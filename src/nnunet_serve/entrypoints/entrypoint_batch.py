"""
Command line utility to perform nnU-Net inference on a multiple studies in SITK
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

from nnunet_serve.api_datamodels import InferenceRequest
from nnunet_serve.nnunet_api import nnUNetAPI
from nnunet_serve.nnunet_api_utils import SUCCESS_STATUS
from nnunet_serve.process_pool import WritingProcessPool
from nnunet_serve.utils import make_parser
from nnunet_serve.logging_utils import get_logger

logger = get_logger(__name__)


def main_with_args(args):
    if args.nproc_writing > 1:
        logger.info("Using %i processes for writing", args.nproc_writing)
        writing_process_pool = WritingProcessPool(args.nproc_writing)
    else:
        writing_process_pool = None
    nnunet_api = nnUNetAPI(writing_process_pool=writing_process_pool)

    responses = []
    with open(args.data_json) as o:
        data = json.load(o)
    logger.info("Processing %i studies", len(data))
    for i, item in enumerate(data):
        logger.info(
            "Processing %s (%i/%i)", item["study_path"], i + 1, len(data)
        )
        inference_request = InferenceRequest(
            nnunet_id=args.nnunet_id,
            study_path=item["study_path"],
            series_folders=item["series_folders"],
            output_dir=item["output_dir"],
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
        all_set_args.extend(list(item.keys()))
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
        response: dict = json.loads(response_obj.body.decode("utf-8"))
        responses.append(response)

    if status_code >= 400 or response.get("status") != SUCCESS_STATUS:
        err = (
            response.get("error") or f"Inference failed with HTTP {status_code}"
        )
        raise RuntimeError(err)
    if writing_process_pool is not None:
        logger.info("Waiting for writing processes to finish")
        writing_responses = {}
        for response in responses:
            out = writing_process_pool.get()
            writing_responses[out[0]] = out[1]
        writing_process_pool.close()
        for response in responses:
            for identifier in writing_responses:
                response.update(writing_responses[identifier])
    for d, response in zip(data, responses):
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
                shutil.copy(
                    final_prediction, os.path.join(d["output_dir"], name)
                )
                response[f"{k}_final"] = name
        del response["metadata"]
    return responses


def main():
    parser = make_parser(
        description="Batch inference for nnunet_serve.",
        exclude=[
            "--study_path",
            "--series_folders",
            "--output_dir",
        ],
    )
    parser.add_argument(
        "--data_json",
        type=str,
        required=True,
        help="Path to data JSON file. This file should contain a list of "
        "dictionaries, each of which has {'study_path': <path>, "
        "'series_folders': <path>, 'output_dir': <path>}.",
    )
    parser.add_argument(
        "--nproc_writing",
        type=int,
        default=1,
        help="Number of processes to use for writing files.",
    )

    args = parser.parse_args()

    pprint.pprint(main_with_args(args))

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
