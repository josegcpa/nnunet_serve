"""
Command line utility to perform nnU-Net inference on a multiple studies in SITK
or DICOM format.
"""

import json
import os
import re
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
from nnunet_serve.logging_utils import get_logger, add_file_handler_to_manager

logger = get_logger(__name__)


def main_with_args(args):
    if args.data_json is None and args.data_dir is None:
        raise ValueError("Must provide either --data_json or --data_dir")
    if args.data_json:
        with open(args.data_json) as o:
            data = json.load(o)
    else:
        if args.output_dir is None:
            raise ValueError("Must provide --output_dir when using --data_dir")
        series_pattern = re.compile(r".*_[0-9]{4}")
        data = []
        for patient_id in Path(args.data_dir).iterdir():
            for study_id in patient_id.iterdir():
                series = [
                    s.name
                    for s in study_id.iterdir()
                    if series_pattern.match(s.name)
                ]
                series = sorted(
                    series,
                    key=lambda x: int(x.split("_")[1].split(".")[0]),
                )
                data.append(
                    {
                        "study_path": str(study_id),
                        "series_folders": [[str(x) for x in series]],
                        "output_dir": str(
                            os.path.join(
                                args.output_dir, patient_id.name, study_id.name
                            )
                        ),
                    }
                )

    print(data)
    add_file_handler_to_manager(
        log_path=os.path.join(data[0]["output_dir"], "nnunet_serve_proc.log"),
        exclude=[
            "nnunet_serve.nnunet_api_utils",
            "nnunet_serve.nnunet_api",
        ],
    )
    if args.nproc_writing > 1:
        logger.info("Using %i processes for writing", args.nproc_writing)
        writing_process_pool = WritingProcessPool(args.nproc_writing)
    else:
        writing_process_pool = None
    nnunet_api = nnUNetAPI(writing_process_pool=writing_process_pool)

    responses = []
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
        "--data_dir",
        type=str,
        default=None,
        help="Path to data directory. This data directory should have a "
        "hierarchical patient/study/series format and each series should be "
        "tagged with an underscore-separated indicator similar to nnU-Net "
        "(e.g. 'series_0000', 'series_0001', etc.). Is overridden by "
        "--data_json. Requires --output_dir to be specified.",
    )
    parser.add_argument(
        "--data_json",
        type=str,
        default=None,
        help="Path to data JSON file. This file should contain a list of "
        "dictionaries, each of which has {'study_path': <path>, "
        "'series_folders': <path>, 'output_dir': <path>}. Overrides --data_dir",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Required when using --data_dir.",
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
