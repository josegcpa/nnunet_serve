"""
Command line utility to perform nnU-Net inference on a single study in SITK
or DICOM format.
"""

import json
import os
import pprint
import shutil
import sys
import asyncio
from pathlib import Path

import torch

from nnunet_serve.nnunet_api import nnUNetAPI
from nnunet_serve.nnunet_api_utils import SUCCESS_STATUS
from nnunet_serve.api_datamodels import InferenceRequest
from nnunet_serve.utils import make_parser
from nnunet_serve.logging_utils import add_file_handler_to_manager


def main_with_args(args):
    add_file_handler_to_manager(
        log_path=os.path.join(args.output_dir, "nnunet_serve.log")
    )
    nnunet_api = nnUNetAPI()

    inference_request = InferenceRequest(
        nnunet_id=args.nnunet_id,
        study_path=args.study_path,
        series_folders=args.series_folders,
        output_dir=args.output_dir,
        class_idx=args.class_idx,
        tmp_dir=args.tmp_dir,
        is_dicom=args.is_dicom,
        use_folds=args.use_folds,
        tta=args.tta,
        proba_threshold=args.proba_threshold,
        intersect_with=args.intersect_with,
        min_intersection=args.min_intersection,
        crop_from=args.crop_from,
        crop_padding=args.crop_padding,
        min_confidence=args.min_confidence,
        cascade_mode=args.cascade_mode,
        save_proba_map=args.proba_map,
        save_nifti_inputs=args.save_nifti_inputs,
        save_rt_struct_output=args.rt_struct_output,
        suffix=args.suffix,
    )

    all_set_args = [k.lstrip("-") for k in sys.argv[1:] if "-" in k]
    # ensures that only the arguments that were set in the CLI are actually used
    inference_request.__pydantic_fields_set__ = [
        k
        for k in inference_request.__pydantic_fields_set__
        if k in all_set_args
    ]

    loop = asyncio.get_event_loop()
    response_obj = loop.run_until_complete(
        asyncio.ensure_future(nnunet_api.infer(inference_request))
    )
    status_code = getattr(response_obj, "status_code", 200)
    response = json.loads(response_obj.body.decode("utf-8"))

    if status_code >= 400 or response.get("status") != SUCCESS_STATUS:
        err = (
            response.get("error") or f"Inference failed with HTTP {status_code}"
        )
        raise RuntimeError(err)
    for k in [
        "nifti_prediction",
        "dicom_segmentation",
        "dicom_struct",
        "dicom_fractional_segmentation",
        "nifti_proba",
    ]:
        if k in response:
            final_prediction = response[k][-1]
            name = Path(final_prediction).name
            shutil.copy(final_prediction, os.path.join(args.output_dir, name))
            response[f"{k}_final"] = name
    del response["metadata"]
    return response


def main():
    parser = make_parser(
        description="Single study inference for nnunet_serve.",
    )

    args = parser.parse_args()

    pprint.pprint(main_with_args(args))

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
