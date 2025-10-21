"""
Implementation of a nnUNet server API. 

Depends on ``model-serve-spec.yaml`` which should be specified in the directory
where nnunet_serve is utilized.
"""

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import fastapi
import numpy as np
import torch
import uvicorn
import yaml
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from nnunet_serve.logging_utils import get_logger
from nnunet_serve.nnunet_serve_utils import (
    FAILURE_STATUS,
    SUCCESS_STATUS,
    InferenceRequest,
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

origins = ["http://localhost:8404"]

logger = get_logger(__name__)


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
        self.app.add_api_route("/infer", self.infer, methods=["POST"])
        self.app.add_api_route("/model_info", self.model_info, methods=["GET"])
        self.app.add_api_route(
            "/request-params", self.request_params, methods=["GET"]
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
                    status_code=400,
                )
            nnunet_info = self.model_dictionary[self.alias_dict[nnunet_id]]
            nnunet_path = nnunet_info["path"]
            min_mem = nnunet_info.get("min_mem", 4000)
            default_args = nnunet_info.get("default_args", {})
            metadata = nnunet_info.get("metadata", None)
        else:
            nnunet_path = []
            metadata = []
            default_args = []
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
                        status_code=400,
                    )
                nnunet_info = self.model_dictionary[self.alias_dict[nn]]
                nnunet_path.append(nnunet_info["path"])
                curr_min_mem = nnunet_info.get("min_mem", 4000)
                if curr_min_mem > min_mem:
                    min_mem = curr_min_mem
                default_args.append(nnunet_info.get("default_args", {}))
                metadata.append(nnunet_info.get("metadata", None))
        default_params = get_default_params(default_args)

        # assign
        for k in default_params:
            if k not in params:
                params[k] = default_params[k]
            else:
                if params[k] is None:
                    params[k] = default_params[k]

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

        device_id = wait_for_gpu(min_mem)

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
                )
                error = None
                status = SUCCESS_STATUS
                torch.cuda.empty_cache()
                b = time.time()

            except Exception as e:
                output_paths = {}
                status = FAILURE_STATUS
                error = str(e)
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
    with open(model_spec_path) as o:
        models_specs = yaml.safe_load(o)
    get_totalseg_dir(models_specs)
    alias_dict = {}
    for model in models_specs["models"]:
        k = model["id"]
        model_name = model["name"]
        alias_dict[model_name] = k
        alias_dict[k] = k
        if "aliases" in model:
            for alias in model["aliases"]:
                alias_dict[alias] = k
            del model["aliases"]
    if "model_folder" not in models_specs:
        raise ValueError(
            "model_folder must be specified in model-serve-spec.yaml"
        )
    grep_str = "|".join([model["rel_path"] for model in models_specs["models"]])
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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return nnunet_api.app


if __name__ == "__main__":
    uvicorn.run(
        "nnunet_serve.nnunet_serve:create_app",
        host="0.0.0.0",
        port=12345,
        reload=True,
    )
