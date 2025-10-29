"""
Implementation of a nnUNet server API. 

Depends on ``model-serve-spec.yaml`` which should be specified in the directory
where nnunet_serve is utilized.
"""

import os
import re
import time
import importlib
from dataclasses import dataclass
from pathlib import Path
from glob import glob

import fastapi
import numpy as np
import torch
import uvicorn
import yaml
from totalsegmentator.map_to_binary import class_map
from fastapi.responses import JSONResponse
from totalsegmentator.libs import download_pretrained_weights
from pydantic import BaseModel, ConfigDict
from typing import Any

from nnunet_serve.totalseg_utils import (
    TASK_CONVERSION,
    load_snomed_mapping_expanded,
)
from nnunet_serve.logging_utils import get_logger
from nnunet_serve.nnunet_serve_utils import (
    FAILURE_STATUS,
    SUCCESS_STATUS,
    InferenceRequest,
    get_gpu_memory,
    get_default_params,
    get_info,
    get_series_paths,
    predict,
    wait_for_gpu,
)

torch.serialization.add_safe_globals(
    [
        np.core.multiarray.scalar,
        np.dtype,
        np.dtypes.Float64DType,
        np.dtypes.Float32DType,
    ]
)

logger = get_logger(__name__)

PORT = int(os.environ.get("NNUNET_SERVE_PORT", "12345"))
TOTAL_SEG_SNOMED_MAPPING = load_snomed_mapping_expanded()


class InferenceResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    time_elapsed: float | None
    gpu: int | None
    nnunet_path: str | list[str] | None
    metadata: Any | None
    request: dict
    status: str
    error: str | None


@dataclass
class nnUNetAPI:
    app: fastapi.FastAPI | None = None

    def __post_init__(self):
        self.model_dictionary, self.alias_dict = get_model_dictionary()

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

    def healthz(self):
        return {"status": "ok"}

    def readyz(self):
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

    def infer(self, inference_request: InferenceRequest):
        """
        Performs inference.

        Args:
            inference_request (InferenceRequest): Inference request.

        Returns:
            JSONResponse: Inference response.
        """
        params = inference_request.__dict__
        nnunet_id = params["nnunet_id"]
        if isinstance(nnunet_id, str):
            if nnunet_id not in self.alias_dict:
                return JSONResponse(
                    content={
                        "time_elapsed": None,
                        "gpu": None,
                        "nnunet_path": None,
                        "metadata": None,
                        "request": inference_request.__dict__,
                        "status": FAILURE_STATUS,
                        "error": f"{nnunet_id} is not a valid nnunet_id",
                    },
                    status_code=404,
                )
            nnunet_info = self.model_dictionary[self.alias_dict[nnunet_id]]
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
                    return JSONResponse(
                        content={
                            "time_elapsed": None,
                            "gpu": None,
                            "nnunet_path": None,
                            "metadata": None,
                            "request": inference_request.__dict__,
                            "status": FAILURE_STATUS,
                            "error": f"{nn} is not a valid nnunet_id",
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
            return JSONResponse(
                content={
                    "time_elapsed": None,
                    "gpu": None,
                    "nnunet_path": None,
                    "metadata": None,
                    "status": FAILURE_STATUS,
                    "request": inference_request.__dict__,
                    "error": "proba_threshold must be not-None if save_proba_map is True",
                },
                status_code=400,
            )

        series_paths, code, error_msg = get_series_paths(
            params["study_path"],
            series_folders=params["series_folders"],
            n=len(nnunet_id) if isinstance(nnunet_id, list) else None,
        )

        if code == FAILURE_STATUS:
            return JSONResponse(
                content={
                    "time_elapsed": None,
                    "gpu": None,
                    "nnunet_path": None,
                    "metadata": None,
                    "status": FAILURE_STATUS,
                    "request": inference_request.__dict__,
                    "error": error_msg,
                },
                status_code=400,
            )

        try:
            device_id = wait_for_gpu(min_mem)
        except (RuntimeError, TimeoutError) as e:
            return JSONResponse(
                content={
                    "time_elapsed": None,
                    "gpu": None,
                    "nnunet_path": None,
                    "metadata": None,
                    "status": FAILURE_STATUS,
                    "request": inference_request.__dict__,
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
            "request": inference_request.__dict__,
            "status": status,
            "error": error,
            **output_paths,
        }
        if status == FAILURE_STATUS:
            return JSONResponse(content=payload, status_code=500)
        return JSONResponse(content=payload, status_code=200)


def get_totalseg_dir(model_specs: dict):
    weights_key = "TOTALSEG_WEIGHTS_PATH"
    if weights_key in os.environ:
        return os.environ[weights_key]
    os.environ[weights_key] = os.path.join(
        model_specs["model_folder"], "totalseg"
    )
    return os.environ[weights_key]


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


def create_app() -> fastapi.FastAPI:
    """
    Creates a FastAPI application.

    Returns:
        fastapi.FastAPI: FastAPI application.
    """
    app = fastapi.FastAPI()
    nnunet_api = nnUNetAPI(app)
    nnunet_api.init_api()

    return nnunet_api.app


if __name__ == "__main__":
    uvicorn.run(
        "nnunet_serve.nnunet_serve:create_app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
    )
