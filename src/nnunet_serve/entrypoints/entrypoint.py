"""
Command line utility to perform nnU-Net inference on a single study in SITK
or DICOM format.
"""

import torch
from nnunet_serve.utils import make_parser
from nnunet_serve.nnunet_serve_utils import InferenceRequest
from nnunet_serve.nnunet_serve import nnUNetAPI


def main(args):
    nnunet_api = nnUNetAPI()

    inference_request = InferenceRequest(
        nnunet_id=args.nnunet_id,
        study_path=args.study_path,
        series_folders=args.series_folders,
        output_dir=args.output_dir,
        class_idx=args.class_idx,
        tmp_dir=args.tmp_dir,
        is_dicom=args.is_dicom,
        use_folds=args.folds,
        tta=args.tta,
        proba_threshold=args.proba_threshold,
        intersect_with=args.intersect_with,
        min_intersection=args.min_intersection,
        crop_from=args.crop_from,
        min_confidence=args.min_confidence,
        cascade_mode=args.cascade_mode,
        save_proba_map=args.proba_map,
        save_nifti_inputs=args.save_nifti_inputs,
        save_rt_struct_output=args.rt_struct_output,
        suffix=args.suffix,
    )

    print(nnunet_api.infer(inference_request).body)


if __name__ == "__main__":
    parser = make_parser()

    args = parser.parse_args()

    args.output_dir = args.output_dir.strip().rstrip("/")
    args.folds = [int(f) for f in args.folds]
    main(args)

    torch.cuda.empty_cache()
