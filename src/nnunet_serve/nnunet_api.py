import datetime
import importlib
import json
import os
import re
import shutil
import sqlite3
import time
import uuid
import asyncio
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

import fastapi
import numpy as np
import torch
import yaml
from fastapi import File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from totalsegmentator.libs import download_pretrained_weights
from totalsegmentator.map_to_binary import class_map

from nnunet_serve.api_datamodels import InferenceRequest, InferenceResponse
from nnunet_serve.file_utils import (
    get_study_path,
    store_uploaded_file,
    zip_directory,
    NNUNET_OUTPUT_DIR,
)
from nnunet_serve.logging_utils import get_logger
from nnunet_serve.nnunet_api_utils import (
    FAILURE_STATUS,
    SUCCESS_STATUS,
    get_default_params,
    get_info,
    get_series_paths,
    predict,
    CACHE,
)
from nnunet_serve.totalseg_utils import (
    TASK_CONVERSION,
    load_snomed_mapping_expanded,
)
from nnunet_serve.utils import get_gpu_memory, wait_for_gpu

logger = get_logger(__name__)


torch.serialization.add_safe_globals(
    [
        np.core.multiarray.scalar,
        np.dtype,
        np.dtypes.Float64DType,
        np.dtypes.Float32DType,
    ]
)

TOTAL_SEG_SNOMED_MAPPING = load_snomed_mapping_expanded()


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
            download_pretrained_weights(task_id)
            matches = glob(os.path.join(totalseg_dir, f"*{task_id}*"))
            if not matches:
                raise FileNotFoundError(
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


@dataclass
class nnUNetAPI:
    app: fastapi.FastAPI | None = None

    def __post_init__(self):
        if torch.cuda.is_available() is False:
            raise ValueError("No GPU available")
        self.model_dictionary, self.alias_dict = get_model_dictionary()
        # Initialise SQLite DB for zip storage
        self._db_path = Path(f"{NNUNET_OUTPUT_DIR}/zip_store.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_conn = sqlite3.connect(self._db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Create the zip_store table if it does not exist.
        Schema: job_id TEXT PRIMARY KEY, created_at DATE, zip_path TEXT
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
        """Insert a new record for a generated zip file.
        ``created_at`` is stored as ISO date (YYYY‑MM‑DD).
        """
        cur = self._db_conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO zip_store (job_id, created_at, zip_path) VALUES (?, ?, ?)",
            (job_id, datetime.date.today().isoformat(), str(zip_path)),
        )
        self._db_conn.commit()

    def _get_zip_path(self, job_id: str) -> Path | None:
        """Retrieve the zip path for ``job_id`` or ``None`` if not found."""
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
            response_model=dict[str, Any],
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
            response_model=dict[str, Any],
        )
        self.app.add_api_route(
            "/request-params",
            self.request_params,
            methods=["GET"],
            response_model=dict[str, Any],
        )
        self.app.add_api_route(
            "/healthz",
            self.healthz,
            methods=["GET"],
            response_model=dict[str, Any],
        )
        self.app.add_api_route(
            "/readyz",
            self.readyz,
            methods=["GET"],
            response_model=dict[str, Any],
        )
        self.app.add_api_route(
            "/expire",
            self.expire,
            methods=["GET"],
            response_model=dict[str, Any],
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
        return self.model_dictionary

    def request_params(self):
        """
        Returns the request parameters.

        Returns:
            dict: Request parameters.
        """
        return InferenceRequest.model_json_schema()

    async def healthz(self):
        return {"status": "ok"}

    async def readyz(self):
        models_loaded = bool(self.model_dictionary)
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

    async def infer(self, inference_request: InferenceRequest):
        """
        Performs inference.

        Args:
            inference_request (InferenceRequest): Inference request.

        Returns:
            JSONResponse: Inference response.
        """
        params = inference_request.__dict__
        if isinstance(params["cascade_mode"], list) is False:
            params["cascade_mode"] = [params["cascade_mode"]]
        params["cascade_mode"] = [x.value for x in params["cascade_mode"]]
        nnunet_id = params["nnunet_id"]
        if isinstance(nnunet_id, str):
            if nnunet_id not in self.alias_dict:
                error_str = f"{nnunet_id} is not a valid nnunet_id"
                if self.app is None:
                    raise ValueError(error_str)
                return JSONResponse(
                    content={
                        "time_elapsed": None,
                        "gpu": None,
                        "nnunet_path": None,
                        "metadata": None,
                        "request": params,
                        "status": FAILURE_STATUS,
                        "error": f"{nnunet_id} is not a valid nnunet_id",
                    },
                    status_code=404,
                )
            nnunet_info: dict = self.model_dictionary[
                self.alias_dict[nnunet_id]
            ]
            nnunet_path = nnunet_info["path"]
            min_mem = nnunet_info.get("min_mem", 4000)
            default_args = nnunet_info.get("default_args", {})
            metadata = nnunet_info.get("metadata", None)
            is_totalseg = nnunet_info.get("is_totalseg", False)
        else:
            nnunet_path = []
            metadata = []
            default_args = []
            is_totalseg = []
            min_mem = 0
            for nn in nnunet_id:
                if nn not in self.alias_dict:
                    error_str = f"{nn} is not a valid nnunet_id"
                    if self.app is None:
                        raise ValueError(error_str)
                    return JSONResponse(
                        content={
                            "time_elapsed": None,
                            "gpu": None,
                            "nnunet_path": None,
                            "metadata": None,
                            "request": params,
                            "status": FAILURE_STATUS,
                            "error": error_str,
                        },
                        status_code=404,
                    )
                nnunet_info = self.model_dictionary[self.alias_dict[nn]]
                nnunet_path.append(nnunet_info["path"])
                curr_min_mem = nnunet_info.get("min_mem", 4000)
                if curr_min_mem > min_mem:
                    min_mem = curr_min_mem
                default_args.append(nnunet_info.get("default_args", {}))
                metadata.append(nnunet_info.get("metadata", None))
                is_totalseg.append(nnunet_info.get("is_totalseg", False))

        default_params = get_default_params(default_args)
        for k in default_params:
            if k not in inference_request.model_fields_set:
                params[k] = default_params[k]
            else:
                if params[k] is None:
                    params[k] = default_params[k]

        if params.get("save_proba_map", False) and all(
            [x is None for x in params.get("proba_threshold", [])]
        ):
            error_str = (
                "proba_threshold must be not-None if save_proba_map is True"
            )
            if self.app is None:
                raise ValueError(error_str)
            return JSONResponse(
                content={
                    "time_elapsed": None,
                    "gpu": None,
                    "nnunet_path": None,
                    "metadata": None,
                    "status": FAILURE_STATUS,
                    "request": params,
                    "error": error_str,
                },
                status_code=400,
            )

        series_paths, code, error_msg = get_series_paths(
            params["study_path"],
            series_folders=params["series_folders"],
            n=len(nnunet_id) if isinstance(nnunet_id, list) else None,
        )

        if code == FAILURE_STATUS:
            error_str = error_msg
            if self.app is None:
                raise ValueError(error_str)
            return JSONResponse(
                content={
                    "time_elapsed": None,
                    "gpu": None,
                    "nnunet_path": None,
                    "metadata": None,
                    "status": FAILURE_STATUS,
                    "request": params,
                    "error": error_msg,
                },
                status_code=400,
            )

        try:
            device_id = wait_for_gpu(min_mem)
        except (RuntimeError, TimeoutError) as e:
            error_str = str(e)
            if self.app is None:
                raise ValueError(error_str)
            return JSONResponse(
                content={
                    "time_elapsed": None,
                    "gpu": None,
                    "nnunet_path": None,
                    "metadata": None,
                    "status": FAILURE_STATUS,
                    "request": params,
                    "error": str(e),
                },
                status_code=503,
            )

        if "tta" in params:
            mirroring = params["tta"]
        else:
            mirroring = True

        a = time.time()
        if os.environ.get("DEBUG", 0) == "1":
            output_paths = predict(
                series_paths=series_paths,
                metadata=metadata,
                mirroring=mirroring,
                device_id=device_id,
                params=params,
                nnunet_path=nnunet_path,
                flip_xy=is_totalseg,
            )
            error = None
            status = SUCCESS_STATUS
        else:
            try:
                output_paths = predict(
                    series_paths=series_paths,
                    metadata=metadata,
                    mirroring=mirroring,
                    device_id=device_id,
                    params=params,
                    nnunet_path=nnunet_path,
                    flip_xy=is_totalseg,
                )
                error = None
                status = SUCCESS_STATUS
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                b = time.time()

            except Exception as e:
                output_paths = {}
                status = FAILURE_STATUS
                error = str(e)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        b = time.time()

        payload = {
            "time_elapsed": b - a,
            "gpu": device_id,
            "nnunet_path": nnunet_path,
            "metadata": metadata,
            "request": params,
            "status": status,
            "error": error,
            **output_paths,
        }
        if status == FAILURE_STATUS:
            error_str = error
            if self.app is None:
                raise ValueError(error_str)
            return JSONResponse(content=payload, status_code=500)
        return JSONResponse(content=payload, status_code=200)

    async def infer_file(
        self,
        inference_request: Request,
        file: UploadFile = File(...),
    ):
        """Accept a file (or archive) upload, stores it, builds an InferenceRequest,
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
                content={
                    "time_elapsed": None,
                    "gpu": None,
                    "nnunet_path": None,
                    "metadata": None,
                    "request": payload,
                    "status": FAILURE_STATUS,
                    "error": f"Invalid request payload: {exc}",
                },
                status_code=422,
            )

        try:
            store_uploaded_file(file, job_id=job_id)
        except Exception as exc:
            return fastapi.responses.JSONResponse(
                content={
                    "time_elapsed": None,
                    "gpu": None,
                    "nnunet_path": None,
                    "metadata": None,
                    "request": {},
                    "status": FAILURE_STATUS,
                    "error": f"Failed to store uploaded file: {exc}",
                },
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

    async def download_file(self, job_id: str):
        """Serve the zip file created for ``job_id``.
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
