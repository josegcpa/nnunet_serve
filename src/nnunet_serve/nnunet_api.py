import datetime
import importlib
import json
import os
import re
import shutil
import sqlite3
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Annotated

import fastapi
import torch
import yaml
from fastapi import File, Request, UploadFile, Query
from fastapi.responses import FileResponse, JSONResponse
from totalsegmentator.libs import download_pretrained_weights
from totalsegmentator.map_to_binary import (
    class_map,
    class_map_5_parts,
    class_map_parts_mr,
    class_map_parts_headneck_muscles,
)

from nnunet_serve.api_datamodels import (
    InferenceRequest,
    InferenceRequestOrthanc,
    InferenceResponse,
    InferenceFileResponse,
    HealthzResponse,
    ReadyzResponse,
    ExpireResponse,
    JSONSchema,
    ModelInfoResponse,
)
from nnunet_serve.file_utils import (
    get_study_path,
    store_uploaded_file,
    zip_directory,
    NNUNET_OUTPUT_DIR,
)
from nnunet_serve.logging_utils import get_logger, add_file_handler_to_manager
from nnunet_serve.nnunet_api_utils import (
    FAILURE_STATUS,
    SUCCESS_STATUS,
    CACHE,
    SAFE_GLOBALS,
    CASCADE_ARGUMENTS,
    get_default_params,
    get_info,
    get_series_paths,
    predict,
)
from nnunet_serve.totalseg_utils import (
    TASK_CONVERSION,
    REVERSE_TASK_CONVERSION,
    load_snomed_mapping_expanded,
)
from nnunet_serve.seg_writers import SegWriter
from nnunet_serve.utils import get_gpu_memory, wait_for_gpu
from nnunet_serve.process_pool import ProcessPool
from nnunet_serve.orthanc_access import download_series, upload_series

logger = get_logger(__name__)

torch.serialization.add_safe_globals(SAFE_GLOBALS)

class_map.update(
    {
        "organ_ct": class_map_5_parts["class_map_part_organs"],
        "vertebrae_ct": class_map_5_parts["class_map_part_vertebrae"],
        "cardiac_ct": class_map_5_parts["class_map_part_cardiac"],
        "muscle_ct": class_map_5_parts["class_map_part_muscles"],
        "ribs_ct": class_map_5_parts["class_map_part_ribs"],
        "organ_mr": class_map_parts_mr["class_map_part_organs"],
        "muscles_mr": class_map_parts_mr["class_map_part_muscles"],
        "headneck_muscles_1": class_map_parts_headneck_muscles[
            "class_map_part_muscles_1"
        ],
        "headneck_muscles_2": class_map_parts_headneck_muscles[
            "class_map_part_muscles_2"
        ],
    }
)

TOTAL_SEG_SNOMED_MAPPING = load_snomed_mapping_expanded()


def make_json_serializable(obj):
    """
    Makes an object JSON serializable by converting Path and nested structures.

    Args:
        obj (Any): The object to make serializable.

    Returns:
        Any: The JSON-serializable version of the object.
    """

    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    return obj


def get_totalseg_dir(model_specs: dict):
    """
    Returns the path to the TotalSegmentator weights directory.

    Args:
        model_specs (dict): The model specifications.

    Returns:
        str: The path to the TotalSegmentator weights directory.
    """
    weights_key = "TOTALSEG_WEIGHTS_PATH"
    if weights_key in os.environ:
        return os.environ[weights_key]

    os.environ[weights_key] = os.path.join(
        model_specs["model_folder"], "totalseg"
    )
    out = os.environ[weights_key]
    Path(out).mkdir(parents=True, exist_ok=True)
    return out


def get_model_dictionary() -> tuple[dict, dict]:
    """
    Returns a dictionary of models and their paths.

    Returns:
        dict, dict: dictionary of models and their paths together with an alias dict.
    """
    model_spec_path = os.environ.get(
        "MODEL_SERVE_SPEC", "model-serve-spec.yaml"
    )
    if not os.path.exists(model_spec_path):
        raise FileNotFoundError(
            f"Model spec file not found at '{model_spec_path}'. Set MODEL_SERVE_SPEC or place model-serve-spec.yaml in CWD."
        )
    try:
        with open(model_spec_path) as o:
            models_specs = yaml.safe_load(o)
    except Exception as e:
        raise RuntimeError(f"Failed to read/parse model spec YAML: {e}") from e
    if not isinstance(models_specs, dict):
        raise ValueError(
            "Model spec must be a YAML mapping/dictionary at top level"
        )
    if "model_folder" not in models_specs or "models" not in models_specs:
        raise ValueError("Model spec must define 'model_folder' and 'models'")
    if (
        not isinstance(models_specs["models"], list)
        or len(models_specs["models"]) == 0
    ):
        raise ValueError("Model spec 'models' must be a non-empty list")
    totalseg_dir = get_totalseg_dir(models_specs)
    alias_dict = {}
    for model in models_specs["models"]:
        k = model["id"]
        model["is_totalseg"] = model.get("is_totalseg", False)
        if "totalseg_task" in model:
            task = model["totalseg_task"]
            task_clean = task.replace("_fastest", "").replace("_fast", "")
            task_id = TASK_CONVERSION[task]
            if isinstance(task_id, list):
                possibilities = [REVERSE_TASK_CONVERSION[t] for t in task_id]
                raise ValueError(
                    "nnunet_serve currently does not support TotalSegmentator multi-part models. "
                    f"Consider using one of the following: {possibilities}."
                    "If multi-part segmentation is important for you: please consider opening an issue "
                    "at https://github.com/josegcpa/nnunet_serve/issues and we will try to push "
                    "this forward."
                )
            else:
                download_pretrained_weights(task_id)
            matches = glob(os.path.join(totalseg_dir, f"*{task_id}*"))
            if not matches:
                raise ValueError(
                    f"Could not find TotalSegmentator weights for task_id '{task_id}' under '{totalseg_dir}'."
                )
            model["rel_path"] = matches[0].replace(
                models_specs["model_folder"], ""
            )
            model["name"] = f"totalseg_{task}"
            segment_names = {v: k for k, v in class_map[task_clean].items()}
            segment_names = sorted(
                segment_names.keys(), key=lambda k: segment_names[k]
            )
            snomed_concepts = [
                {
                    "name": k,
                    **TOTAL_SEG_SNOMED_MAPPING[k]["property_type"],
                    "label": TOTAL_SEG_SNOMED_MAPPING[k]["property_type"][
                        "meaning"
                    ],
                }
                for k in segment_names
            ]
            model["is_totalseg"] = True
            model["metadata"] = {
                "segment_names": snomed_concepts,
                "algorithm_name": "TotalSegmentator",
                "algorithm_version": (
                    importlib.metadata.version("TotalSegmentator")
                    if hasattr(importlib, "metadata")
                    else "unknown"
                ),
                "manufacturer": "TotalSegmentator",
                "manufacturer_model_name": "TotalSegmentator",
                "series_description": f"TotalSegmentator (task: {task})",
                "body_part_examined": "BODY",
            }

        model_name = model["name"]
        alias_dict[model_name] = k
        alias_dict[k] = k
        if "aliases" in model:
            for alias in model["aliases"]:
                alias_dict[alias] = k
            del model["aliases"]
    grep_str = "|".join(
        [re.escape(model["rel_path"]) for model in models_specs["models"]]
    )
    pat = re.compile(grep_str)

    model_folder = models_specs["model_folder"]
    model_paths = [
        os.path.dirname(x) for x in Path(model_folder).rglob("fold_0")
    ]
    model_dictionary = {}
    for m in model_paths:
        match = pat.search(m)
        if match is not None:
            match = match.group()
            model_dictionary[match] = {
                "path": m,
                "model_information": get_info(f"{m}/dataset.json"),
            }
            model_dictionary[match]["n_classes"] = len(
                model_dictionary[match]["model_information"]["labels"]
            )

    output_model_dictionary = {}
    for m in model_dictionary:
        model_spec = [
            model for model in models_specs["models"] if model["rel_path"] == m
        ]
        if len(model_spec) == 0:
            continue
        model_spec = model_spec[0]
        k = model_spec["id"]
        output_model_dictionary[k] = model_dictionary[m]
        output_model_dictionary[k].update(model_spec)
    for k in output_model_dictionary:
        logger.debug("Model dictionary: %s=%s", k, output_model_dictionary[k])
    for k in alias_dict:
        logger.debug("Alias dictionary: %s=%s", k, alias_dict[k])
    return output_model_dictionary, alias_dict


def dict_to_str(d: dict) -> str:
    """
    Converts a dictionary to a string for display.

    Args:
        d (dict): Dictionary which will be converted to string.

    Returns:
        str: Stringified version of the dictionary.
    """

    return ", ".join([f"{k}: {v}" for k, v in d.items()])


def normalize_inference_params(inference_request: InferenceRequest) -> dict:
    """Normalize request payload into the dict format consumed by inference.

    Args:
        inference_request (InferenceRequest): Parsed inference request model.

    Returns:
        dict: Mutable dictionary with enum-like fields converted to plain values.
    """
    params = inference_request.__dict__
    if isinstance(params["cascade_mode"], list) is False:
        params["cascade_mode"] = [params["cascade_mode"]]
    params["cascade_mode"] = [x.value for x in params["cascade_mode"]]
    params["checkpoint_name"] = [x.value for x in params["checkpoint_name"]]
    return params


def expand_cascade_inputs(
    params: dict,
    nnunet_id: list[str],
    model_dictionary: dict,
    alias_dict: dict,
    may_inject_series: list[dict] | None = None,
) -> tuple[list[str], list[tuple[int, str]]]:
    """Expand ``from:<model>`` references into explicit cascade stages.

    This function mutates ``params`` and ``nnunet_id`` in place to inject missing
    upstream cascade stages required by ``from:`` references in ``series_folders``.

    Args:
        params (dict): Normalized request parameters.
        nnunet_id (list[str]): Requested model identifiers by stage.
        model_dictionary (dict): Model metadata indexed by canonical id.
        alias_dict (dict): Alias-to-canonical-id mapping.
        may_inject_series (list[dict], optional): Optional list of dictionaries
            indicating which series in each cascade stage may need to be injected
            from upstream model outputs. Each dictionary maps series indices to
            their "from:" reference strings. When provided, enables automatic
            injection of missing upstream cascade stages based on these
            references. Defaults to None.

    Returns:
        tuple[list[str], list[tuple[int, str]]]:
            - Updated stage-ordered ``nnunet_id`` list.
            - Insertion metadata as ``(index, model_id)`` tuples.
    """
    if may_inject_series is None:
        may_inject_series = [{} for _ in nnunet_id]
    new_inputs = []
    insert_at = []
    for idx, _ in enumerate(nnunet_id):
        series_ids = params["series_folders"][idx]
        if len(may_inject_series[idx]) > 0:
            for k in may_inject_series[idx]:
                if len(series_ids) <= k:
                    series_ids.append(may_inject_series[idx][k])
        for sid_idx, sid in enumerate(series_ids):
            if sid.startswith("from:"):
                prev_stage_sid = sid.split(":")[1]
                is_equal = "=" in prev_stage_sid
                is_index = "[" in prev_stage_sid
                if is_equal:
                    prev_stage_nnunet_id, pred_id = prev_stage_sid.split("=")
                elif is_index:
                    prev_stage_nnunet_id, pred_id = prev_stage_sid.split("[")
                    pred_id = pred_id.replace("]", "")
                else:
                    prev_stage_nnunet_id, pred_id = prev_stage_sid, None

                pred_name = "prediction.nii.gz"
                if is_equal:
                    pred_name = f"prediction.nii.gz={pred_id}"
                if is_index:
                    pred_name = f"prediction.nii.gz[{pred_id}]"
                if prev_stage_nnunet_id not in nnunet_id[:idx]:
                    ins = (idx, prev_stage_nnunet_id)
                    if ins not in insert_at:
                        channels = model_dictionary[
                            alias_dict[prev_stage_nnunet_id]
                        ]["model_information"]["channel_names"]
                        new_inputs.append(series_ids[: len(channels)])
                        insert_at.append(ins)
                    series_ids[sid_idx] = Path(
                        os.path.join(f"stage_{idx}", pred_name)
                    )
                else:
                    stage_idx = nnunet_id[:idx].index(prev_stage_nnunet_id)
                    series_ids[sid_idx] = Path(
                        os.path.join(f"stage_{stage_idx}", pred_name)
                    )

    for i in range(len(insert_at)):
        idx, prev_stage_nnunet_id = insert_at[i]
        nnunet_id.insert(idx, prev_stage_nnunet_id)
        params["series_folders"].insert(idx, new_inputs[i])
        for k in CASCADE_ARGUMENTS:
            if k == "series_folders":
                continue
            if k in params and isinstance(params[k], list):
                params[k].insert(idx, None)
    return nnunet_id, insert_at


def resolve_models(
    nnunet_id: list[str], model_dictionary: dict, alias_dict: dict
) -> tuple[list[str], list[Any], list[dict], list[bool], int, str | None]:
    """
    Resolve model ids to paths/metadata and compute shared execution config.

    Args:
        nnunet_id (list[str]): Stage-ordered requested model ids.
        model_dictionary (dict): Model metadata indexed by canonical id.
        alias_dict (dict): Alias-to-canonical-id mapping.

    Returns:
        tuple[list[str], list[Any], list[dict], list[bool], int, str | None]:
            - Resolved model paths.
            - Per-model output metadata.
            - Per-model default argument dictionaries.
            - Per-model TotalSegmentator flags.
            - Maximum required free GPU memory across models.
            - Error string when resolution fails, otherwise ``None``.
    """

    nnunet_path = []
    metadata = []
    default_args = []
    is_totalseg = []
    min_mem = 0
    for nn in nnunet_id:
        if nn not in alias_dict:
            return [], [], [], [], 0, f"{nn} is not a valid nnunet_id"
        nnunet_info = model_dictionary[alias_dict[nn]]
        nnunet_path.append(nnunet_info["path"])
        curr_min_mem = nnunet_info.get("min_mem", 4000)
        if curr_min_mem > min_mem:
            min_mem = curr_min_mem
        default_args.append(nnunet_info.get("default_args", {}))
        metadata.append(nnunet_info.get("metadata", None))
        is_totalseg.append(nnunet_info.get("is_totalseg", False))
    return nnunet_path, metadata, default_args, is_totalseg, min_mem, None


def get_may_inject(default_args: list[dict]):
    """
    Determines whether there are injectable series. These are exclusively the
    series which are to be obtained from an inference (i.e. starting with a `from:`).

    Args:
        default_args (list[dict]): Default argument dictionaries from model specs.

    Returns:
        list[dict]: List of dictionaries where each dictionary contains the index of the series
        that is to be injected if necessary.
    """
    default_params = get_default_params(default_args)
    if "series_folders" in default_params:
        all_may_inject = []
        series_folders = default_params["series_folders"]
        for sf in series_folders:
            may_inject = {}
            for j, s in enumerate(sf):
                if "from:" in s:
                    may_inject[j] = s
            all_may_inject.append(may_inject)
        return all_may_inject
    return None


def apply_request_defaults(
    params: dict,
    default_args: list[dict],
    inference_request: InferenceRequest,
    insert_at: list[tuple[int, str]],
) -> None:
    """
    Apply model-level default args to normalized request params.

    Args:
        params (dict): Normalized and expanded request parameters.
        default_args (list[dict]): Default argument dictionaries from model specs.
        inference_request (InferenceRequest): Original request model used to detect
            explicitly set fields.
        insert_at (list[tuple[int, str]]): Cascade insertion metadata from
            ``expand_cascade_inputs``.
    """
    default_params = get_default_params(default_args)
    for k in default_params:
        set_to_default = False
        if k not in inference_request.model_fields_set:
            set_to_default = True
        elif params[k] is None:
            set_to_default = True
        if set_to_default:
            params[k] = default_params[k]
        elif k in CASCADE_ARGUMENTS and k != "series_folders" and insert_at:
            for ins in insert_at:
                v = None
                if k in default_params:
                    v = default_params[k][ins[0]]
                params[k][ins[0]] = v


def run_predict_inference(
    *,
    series_paths: list,
    metadata: list[Any],
    mirroring: bool,
    params: dict,
    nnunet_path: list[str],
    is_totalseg: list[bool],
    writing_process_pool: ProcessPool | None,
) -> tuple[dict, list[str], list[bool], str, str | None]:
    """
    Run ``predict`` and normalize execution result into a common shape.

    Args:
        series_paths (list): Stage-wise input paths passed to ``predict``.
        metadata (list[Any]): Stage-wise metadata payload passed to ``predict``.
        mirroring (bool): Whether test-time mirroring is enabled.
        params (dict): Effective normalized inference parameters.
        nnunet_path (list[str]): Stage-wise model paths.
        is_totalseg (list[bool]): Stage-wise TotalSegmentator flags.
        writing_process_pool (ProcessPool | None): Optional process pool for
            asynchronous export writes.

    Returns:
        tuple[dict, list[str], list[bool], str, str | None]:
            ``(output_paths, identifiers, is_empty, status, error)`` where
            ``status`` is either ``SUCCESS_STATUS`` or ``FAILURE_STATUS``.
    """
    if os.environ.get("DEBUG", "0") == "1":
        output_paths, identifiers, is_empty = predict(
            series_paths=series_paths,
            metadata=metadata,
            mirroring=mirroring,
            device_id=None,
            params=params,
            nnunet_path=nnunet_path,
            flip_xy=is_totalseg,
            writing_process_pool=writing_process_pool,
        )
        status = SUCCESS_STATUS
        error = None
    else:
        try:
            output_paths, identifiers, is_empty = predict(
                series_paths=series_paths,
                metadata=metadata,
                mirroring=mirroring,
                device_id=None,
                params=params,
                nnunet_path=nnunet_path,
                flip_xy=is_totalseg,
                writing_process_pool=writing_process_pool,
            )
            status = SUCCESS_STATUS
            error = None
        except Exception as e:
            output_paths = {}
            identifiers = []
            is_empty = []
            status = FAILURE_STATUS
            error = str(e)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_paths, identifiers, is_empty, status, error


def build_infer_success_payload(
    *,
    time_elapsed: float,
    nnunet_path: list[str],
    metadata: list[Any],
    request_params: dict,
    identifiers: list[str],
    is_empty: list[bool],
    output_paths: dict,
) -> dict[str, Any]:
    """Build the canonical success payload returned by ``infer``.

    Args:
        time_elapsed (float): Inference wall-clock time in seconds.
        nnunet_path (list[str]): Resolved model path(s) used in inference.
        metadata (list[Any]): Metadata used for export operations.
        request_params (dict): Effective request parameters used in execution.
        identifiers (list[str]): Async export identifiers (if applicable).
        is_empty (list[bool]): Per-stage empty-mask flags.
        output_paths (dict): Export artifact paths.

    Returns:
        dict[str, Any]: JSON-serializable success payload.
    """
    payload = {
        "time_elapsed": time_elapsed,
        "nnunet_path": nnunet_path,
        "metadata": metadata,
        "request": request_params,
        "status": SUCCESS_STATUS,
        "error": None,
        "identifiers": identifiers,
        "is_empty": is_empty,
        **output_paths,
    }
    return make_json_serializable(payload)


@dataclass
class nnUNetAPI:
    app: fastapi.FastAPI | None = None
    writing_process_pool: ProcessPool | None = None
    """
    General nnU-Net API.
    
    **Important:** the ``writing_process_pool`` is only implemented for the 
    command line entrypoints and, as such, is only used when ``app`` is None.
    
    Args:
        app: FastAPI application
        writing_process_pool: ProcessPool for writing files.
    """

    def __post_init__(self):
        """
        Initializes the nnUNetAPI instance, checking for GPU and setting up the DB.
        """
        if torch.cuda.is_available() is False:
            raise ValueError("No GPU available")
        self.model_dictionary, self.alias_dict = get_model_dictionary()
        # Initialise SQLite DB for zip storage
        self._db_path = Path(f"{NNUNET_OUTPUT_DIR}/zip_store.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_conn = sqlite3.connect(self._db_path)
        self._init_db()

    def _init_db(self) -> None:
        """
        Creates the zip_store table if it does not exist.

        Schema:
            job_id (TEXT): Primary key.
            created_at (DATE): ISO date of creation.
            zip_path (TEXT): Path to the generated zip file.
        """
        cur = self._db_conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS zip_store (
                job_id TEXT PRIMARY KEY,
                created_at DATE,
                zip_path TEXT
            )
            """
        )
        self._db_conn.commit()

    def _store_zip(self, job_id: str, zip_path: Path) -> None:
        """Inserts or replaces a record for a generated zip file.

        Args:
            job_id (str): Unique job identifier.
            zip_path (Path): Path to the generated zip file.
        """
        cur = self._db_conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO zip_store (job_id, created_at, zip_path) VALUES (?, ?, ?)",
            (job_id, datetime.date.today().isoformat(), str(zip_path)),
        )
        self._db_conn.commit()

    def _get_zip_path(self, job_id: str) -> Path | None:
        """
        Retrieves the zip path for a given job_id.

        Args:
            job_id (str): Unique job identifier.

        Returns:
            Path | None: The path to the zip file if found, else None.
        """
        cur = self._db_conn.cursor()
        cur.execute(
            "SELECT zip_path FROM zip_store WHERE job_id = ?", (job_id,)
        )
        row = cur.fetchone()
        return Path(row[0]) if row else None

    def cleanup_old_records(self, days: int = 7) -> int:
        """
        Delete records older than *days*.
        Returns the number of rows removed.

        Args:
            days (int, optional): Number of days to keep records. Defaults to 7.
        """
        cutoff = datetime.date.today() - datetime.timedelta(days=days)
        cur = self._db_conn.cursor()
        cur.execute(
            "DELETE FROM zip_store WHERE created_at < ?", (cutoff.isoformat(),)
        )
        removed = cur.rowcount
        self._db_conn.commit()
        return removed

    def init_api(self):
        """
        Initializes the API.
        """
        if self.app is None:
            raise ValueError("app must be defined before init_api is called")
        self.app.add_api_route(
            "/infer",
            self.infer,
            methods=["POST"],
            response_model=InferenceResponse,
        )
        self.app.add_api_route(
            "/infer_file",
            self.infer_file,
            methods=["POST"],
            response_model=InferenceFileResponse,
        )
        self.app.add_api_route(
            "/infer_orthanc",
            self.infer_orthanc,
            methods=["POST"],
            response_model=InferenceResponse,
        )
        self.app.add_api_route(
            "/download/{job_id}",
            self.download_file,
            methods=["GET"],
            response_class=FileResponse,
        )
        self.app.add_api_route(
            "/model_info",
            self.model_info,
            methods=["GET"],
            response_model=ModelInfoResponse,
        )
        self.app.add_api_route(
            "/model_info_clean",
            self.model_info_clean,
            methods=["GET"],
            response_model=ModelInfoResponse,
        )
        self.app.add_api_route(
            "/request-params",
            self.request_params,
            methods=["GET"],
            response_model=JSONSchema,
        )
        self.app.add_api_route(
            "/healthz",
            self.healthz,
            methods=["GET"],
            response_model=HealthzResponse,
        )
        self.app.add_api_route(
            "/readyz",
            self.readyz,
            methods=["GET"],
            response_model=ReadyzResponse,
        )
        self.app.add_api_route(
            "/expire",
            self.expire,
            methods=["GET"],
            response_model=ExpireResponse,
        )

    def expire(self):
        """
        Calls the TTL cache expire method.
        """
        n_items = 0
        try:
            n_items = len(CACHE.expire())
        except Exception as e:
            logger.error("Failed to expire cache: %s", e)
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": str(e)},
            )
        return JSONResponse(
            status_code=200,
            content={"status": "ok", "message": f"Expired {n_items} items"},
        )

    def model_info(self):
        """
        Returns the model information.

        Returns:
            dict: Model information.
        """
        return self.model_info_clean()

    def model_info_clean(self):
        """
        Returns the model information with cleaned metadata.

        Returns:
            dict: Model information.
        """
        model_dict = deepcopy(self.model_dictionary)
        for model in model_dict.values():
            sd = SegWriter.init_from_metadata_dict(
                model["metadata"]
            ).segment_descriptions
            model_labels = {
                v: k for k, v in model["model_information"]["labels"].items()
            }
            for i in range(len(sd)):
                try:
                    label = sd[i][0x0062, 0x0005].value
                except KeyError:
                    label = None
                try:
                    meaning = sd[i][0x0062, 0x000F][0][0x0008, 0x0104].value
                except KeyError:
                    meaning = None
                try:
                    laterality = sd[i][0x0062, 0x0011][0][0x0008, 0x0104].value
                except KeyError:
                    laterality = None
                sd[i] = {
                    "Label ID": model_labels[i + 1],
                    "Name": label,
                    "Index": i + 1,
                }

            model["metadata"] = sd
            model["description"] = "\n".join(
                [
                    "Segments the following regions:",
                    ", ".join([model_labels[i + 1] for i in range(len(sd))]),
                    "Number of input channels:",
                    str(len(model["model_information"]["channel_names"])),
                ]
            )
            model["description_long"] = "\n".join(
                [
                    "Segments the following regions:",
                    *["\t- " + dict_to_str(sd[i]) for i in range(len(sd))],
                    "Uses the following channels:",
                    "\t- "
                    + dict_to_str(model["model_information"]["channel_names"]),
                ]
            )
        return model_dict

    def request_params(self):
        """
        Returns the request parameters.

        Returns:
            dict: Request parameters.
        """
        return InferenceRequest.model_json_schema()

    def _failure_payload(
        self,
        error: str,
        request_payload: dict | None,
        **extra_fields,
    ) -> dict[str, Any]:
        """
        Build a standardized failure payload for API responses.

        Args:
            error (str): Error message.
            request_payload (dict | None): Request payload.
            **extra_fields: Extra fields to add to the payload.

        Returns:
            dict[str, Any]: Failure payload.
        """
        payload = {
            "time_elapsed": None,
            "gpu": None,
            "nnunet_path": None,
            "metadata": None,
            "status": FAILURE_STATUS,
            "request": request_payload if request_payload is not None else {},
            "error": error,
        }
        payload.update(extra_fields)
        return payload

    def _raise_or_error_response(
        self,
        error: str,
        status_code: int,
        request_payload: dict | None,
        exception_type: type[Exception] = ValueError,
        **extra_fields,
    ):
        """
        Raise in CLI mode or return standardized JSON error in API mode.

        Args:
            error (str): Error message.
            status_code (int): status code to be used for error response.
            request_payload (dict | None): payload for the request.
            exception_type (type[Exception] = ValueError): type of Exception.
            **extra_fields: Extra fields to add to the payload.

        Returns:
            JSONResponse if self.app is not None.

        Raises:
            ``exception_type`` error.
        """
        if self.app is None:
            raise exception_type(error)
        return JSONResponse(
            content=self._failure_payload(
                error=error,
                request_payload=request_payload,
                **extra_fields,
            ),
            status_code=status_code,
        )

    async def healthz(self):
        """
        Returns a simple health check.
        """
        return {"status": "ok"}

    async def readyz(self):
        """
        Returns a readiness check.
        """
        models_loaded = len(self.model_dictionary) > 0
        gpu_available = False
        max_free_mem = None
        try:
            if torch.cuda.is_available():
                mem = get_gpu_memory()
                gpu_available = len(mem) > 0
                max_free_mem = max(mem) if mem else None
        except Exception:
            gpu_available = False
        status = (
            "ok"
            if models_loaded
            and (gpu_available or not torch.cuda.is_available())
            else "starting"
        )
        return {
            "status": status,
            "models_loaded": models_loaded,
            "gpu_available": gpu_available,
            "max_free_mem": max_free_mem,
        }

    async def infer(
        self, inference_request: Annotated[InferenceRequest, Query()]
    ):
        """
        Performs inference.

        Args:
            inference_request (InferenceRequest): Inference request.

        Returns:
            JSONResponse: Inference response.
        """
        if self.app is not None and self.writing_process_pool is not None:
            raise ValueError("Cannot use both app and writing_process_pool")
        params = normalize_inference_params(inference_request)
        add_file_handler_to_manager(
            log_path=os.path.join(params["output_dir"], "nnunet_serve.log"),
            exclude=[
                "nnunet_serve.entrypoints.entrypoint_batch",
                "nnunet_serve.entrypoints.entrypoint",
                "nnunet_serve.process_pool",
                "nnunet_serve.seg_writers",
            ],
        )
        nnunet_id = params["nnunet_id"]
        if isinstance(nnunet_id, str):
            nnunet_id = [nnunet_id]

        initial_default_args = resolve_models(
            nnunet_id=nnunet_id,
            model_dictionary=self.model_dictionary,
            alias_dict=self.alias_dict,
        )[2]
        may_inject = get_may_inject(initial_default_args)

        nnunet_id, insert_at = expand_cascade_inputs(
            params=params,
            nnunet_id=nnunet_id,
            model_dictionary=self.model_dictionary,
            alias_dict=self.alias_dict,
            may_inject_series=may_inject,
        )

        (
            nnunet_path,
            metadata,
            default_args,
            is_totalseg,
            min_mem,
            model_resolution_error,
        ) = resolve_models(
            nnunet_id=nnunet_id,
            model_dictionary=self.model_dictionary,
            alias_dict=self.alias_dict,
        )
        if model_resolution_error is not None:
            return self._raise_or_error_response(
                error=model_resolution_error,
                status_code=404,
                request_payload=params,
            )

        apply_request_defaults(
            params=params,
            default_args=default_args,
            inference_request=inference_request,
            insert_at=insert_at,
        )
        params["min_mem"] = min_mem

        if params.get("save_proba_map", False) and all(
            [x is None for x in params.get("proba_threshold", [])]
        ):
            error_str = (
                "proba_threshold must be not-None if save_proba_map is True"
            )
            return self._raise_or_error_response(
                error=error_str,
                status_code=400,
                request_payload=params,
            )

        series_paths, code, error_msg = get_series_paths(
            params["study_path"],
            series_folders=params["series_folders"],
            n=len(nnunet_id) if isinstance(nnunet_id, list) else None,
        )

        if code == FAILURE_STATUS:
            error_str = error_msg
            return self._raise_or_error_response(
                error=error_str,
                status_code=400,
                request_payload=params,
            )

        try:
            wait_for_gpu(min_mem)
        except (RuntimeError, TimeoutError) as e:
            error_str = str(e)
            return self._raise_or_error_response(
                error=error_str,
                status_code=503,
                request_payload=params,
                exception_type=RuntimeError,
            )

        if "tta" in params:
            mirroring = params["tta"]
        else:
            mirroring = True

        a = time.time()
        (
            output_paths,
            identifiers,
            is_empty,
            status,
            error,
        ) = run_predict_inference(
            series_paths=series_paths,
            metadata=metadata,
            mirroring=mirroring,
            params=params,
            nnunet_path=nnunet_path,
            is_totalseg=is_totalseg,
            writing_process_pool=self.writing_process_pool,
        )
        b = time.time()
        if status == FAILURE_STATUS:
            error_str = error
            return self._raise_or_error_response(
                error=error_str,
                status_code=500,
                request_payload=params,
                identifiers=identifiers,
                is_empty=is_empty,
                **output_paths,
            )
        payload = build_infer_success_payload(
            time_elapsed=b - a,
            nnunet_path=nnunet_path,
            metadata=metadata,
            request_params=params,
            identifiers=identifiers,
            is_empty=is_empty,
            output_paths=output_paths,
        )
        return JSONResponse(content=payload, status_code=200)

    async def infer_file(
        self,
        inference_request: Request,
        file: UploadFile = File(...),
    ):
        """
        Accept a file (or archive) upload, stores it, builds an InferenceRequest,
        and delegates to the existing ``infer`` method.
        """

        job_id = uuid.uuid4().hex
        form = await inference_request.form()
        json_str = form.get("request")
        if json_str is not None:
            payload = json.loads(json_str)
        else:
            payload = await inference_request.json()

        study_path = get_study_path(job_id)
        payload["study_path"] = str(study_path / "inputs")
        payload["output_dir"] = str(study_path / "output")

        try:
            inference_req = InferenceRequest(**payload)
        except Exception as exc:
            return fastapi.responses.JSONResponse(
                content=self._failure_payload(
                    error=f"Invalid request payload: {exc}",
                    request_payload=payload,
                ),
                status_code=422,
            )

        try:
            store_uploaded_file(file, job_id=job_id)
        except Exception as exc:
            return fastapi.responses.JSONResponse(
                content=self._failure_payload(
                    error=f"Failed to store uploaded file: {exc}",
                    request_payload={},
                ),
                status_code=400,
            )

        response = await self.infer(inference_req)

        if response.status_code == 200:
            zip_path = zip_directory(Path(inference_req.output_dir))
            self._store_zip(job_id, zip_path)
            shutil.rmtree(inference_req.study_path)
            shutil.rmtree(inference_req.output_dir)
            original = json.loads(response.body)
            original.update({"job_id": job_id})
            return JSONResponse(content=original, status_code=200)
        else:
            error_payload = json.loads(response.body)
            error_payload.update({"job_id": job_id})
            return JSONResponse(
                content=error_payload, status_code=response.status_code
            )

    async def infer_orthanc(
        self, inference_request: Annotated[InferenceRequestOrthanc, Query()]
    ):
        """Run inference for Orthanc-backed inputs and push SEG back to Orthanc.

        This adapter:
        1. Downloads Orthanc series referenced in ``series_ids``.
        2. Rewrites ``series_folders`` to local downloaded paths.
        3. Reuses ``infer``.
        4. Uploads produced DICOM SEG files back to Orthanc.

        Entries using ``from:`` are preserved as-is to keep cascade behavior.
        """
        job_id = uuid.uuid4().hex
        study_path = get_study_path(job_id)
        inputs_path = study_path / "inputs"
        output_path = study_path / "output"
        inputs_path.mkdir(parents=True, exist_ok=True)

        payload = inference_request.model_dump()
        payload["study_path"] = str(inputs_path)
        payload["output_dir"] = str(output_path)

        raw_series_ids = payload.get("series_ids", None)
        if raw_series_ids is None:
            return self._raise_or_error_response(
                error="series_ids must be defined",
                status_code=400,
                request_payload=payload,
            )

        if isinstance(raw_series_ids, list) and (
            len(raw_series_ids) == 0 or isinstance(raw_series_ids[0], str)
        ):
            series_ids = [raw_series_ids]
        else:
            series_ids = raw_series_ids

        orthanc_series_ids = []
        for stage_series in series_ids:
            for sid in stage_series:
                if isinstance(sid, str) and sid.startswith("from:"):
                    continue
                orthanc_series_ids.append(sid)

        local_series_map = {}
        if len(orthanc_series_ids) > 0:
            unique_series_ids = sorted(set(orthanc_series_ids))
            downloaded_paths = download_series(
                unique_series_ids, output_dir=str(inputs_path)
            )
            for sid, folder_path in downloaded_paths.items():
                local_series_map[sid] = os.path.relpath(
                    folder_path, inputs_path
                )

        adapted_series_folders = []
        for stage_series in series_ids:
            adapted_stage = []
            for sid in stage_series:
                if isinstance(sid, str) and sid.startswith("from:"):
                    adapted_stage.append(sid)
                else:
                    adapted_stage.append(local_series_map[sid])
            adapted_series_folders.append(adapted_stage)

        payload.pop("series_ids", None)
        payload["series_folders"] = adapted_series_folders

        try:
            infer_request = InferenceRequest(**payload)
        except Exception as exc:
            shutil.rmtree(study_path, ignore_errors=True)
            return self._raise_or_error_response(
                error=f"Invalid adapted request payload: {exc}",
                status_code=422,
                request_payload=payload,
            )

        response = await self.infer(infer_request)
        if response.status_code != 200:
            shutil.rmtree(study_path, ignore_errors=True)
            return response

        response_payload = json.loads(response.body)
        dicom_seg_paths = response_payload.get("dicom_segmentation", [])
        dicom_seg_paths = [p for p in dicom_seg_paths if p is not None]

        uploaded_instances = []
        if len(dicom_seg_paths) > 0:
            uploaded_instances = upload_series(dicom_seg_paths)
        response_payload["orthanc_upload"] = {
            "uploaded_instance_count": len(uploaded_instances),
            "responses": uploaded_instances,
        }

        shutil.rmtree(study_path, ignore_errors=True)
        return JSONResponse(content=response_payload, status_code=200)

    async def download_file(self, job_id: str):
        """
        Serve the zip file created for ``job_id`` (``job_id`` is the value returned by
        the ``infer_file`` endpoint).
        Returns 404 if the ``job_id`` is unknown or the file has been cleaned up.
        """
        zip_path = self._get_zip_path(job_id)
        if not zip_path or not zip_path.exists():
            raise fastapi.HTTPException(
                status_code=404, detail="Zip not found for job_id"
            )
        return FileResponse(
            path=zip_path, media_type="application/zip", filename=zip_path.name
        )

    def __del__(self):
        if hasattr(self, "_db_conn"):
            self._db_conn.close()
