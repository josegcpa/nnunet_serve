import argparse


def float_or_none(x: str) -> float | None:
    """
    Converts a string to float if it is different from "none"; if that is the
    case, the function returns None.

    Args:
        x (str): string to be converted.

    Returns:
        float | None: None or the converted float.
    """
    if x.lower() == "none":
        return None
    return float(x)


def int_or_list_of_ints(x: str) -> int | list[int]:
    """
    Converts a comma-separated string of integers into an int or list.

    The input string is first stripped of spaces and trailing commas. If the
    value is "all" (case-insensitive), the function returns ``None``. If the
    string contains commas, it is split into a list of unique integers.
    Otherwise, it is converted to a single integer.

    Args:
        x (str): String encoding an integer, a comma-separated list of
            integers, or the special value ``"all"``.

    Returns:
        int | list[int] | None: ``None`` for ``"all"``, a single integer, or a
        list of unique integers.
    """
    x = x.replace(" ", "").strip(",")
    if x.lower() == "all":
        return None
    if "," in x:
        return list(set([int(y) for y in x.split(",")]))
    return int(x)


def int_or_float(x: str) -> int | float:
    """
    Parses a string into an integer when possible, otherwise a float.

    Args:
        x (str): String representation of a numeric value.

    Returns:
        int | float: The parsed integer if conversion to ``int`` succeeds,
        otherwise the parsed ``float``.
    """
    try:
        return int(x)
    except ValueError:
        return float(x)


def list_of_str(x: str) -> list[str]:
    """Splits a comma-separated string into a list of strings.

    Args:
        x (str): Comma-separated string.

    Returns:
        list[str]: List of substrings obtained by splitting on commas.
    """
    return x.split(",")


def make_parser(
    description: str = "Entrypoint for nnUNet prediction. Handles all data "
    "format conversions and cascades of predictions.",
    exclude: list[str] = None,
) -> argparse.ArgumentParser:
    """
    Convenience function to generate ``argparse`` CLI parser. Helps with
    consistent inputs when dealing with multiple entrypoints.

    Args:
        description (str, optional): description for the
            ``argparse.ArgumentParser`` call. Defaults to "Entrypoint for nnUNet
            prediction. Handles all data format conversions.".

    Returns:
        argparse.ArgumentParser: parser with specific args.
    """
    parser = argparse.ArgumentParser(description)
    if exclude is None:
        exclude = []
    args = [
        (
            ("--study_path", "-i"),
            {"help": "Path to input series", "required": True},
        ),
        (
            ("--series_folders", "-s"),
            {
                "nargs": "+",
                "type": list_of_str,
                "help": "Path to input series folders",
                "required": True,
            },
        ),
        (
            ("--nnunet_id",),
            {
                "nargs": "+",
                "help": "nnUNet ID",
                "required": True,
            },
        ),
        (
            ("--checkpoint_name",),
            {
                "help": "Checkpoint name for nnUNet",
                "default": "checkpoint_final.pth",
                "nargs": "+",
            },
        ),
        (
            ("--output_dir", "-o"),
            {"help": "Path to output directory", "required": True},
        ),
        (
            ("--use_folds", "-f"),
            {
                "help": "Sets which folds should be used with nnUNet",
                "nargs": "+",
                "type": int_or_list_of_ints,
                "default": (0,),
            },
        ),
        (
            ("--tta", "-t"),
            {
                "help": "Uses test-time augmentation during prediction",
                "action": "store_true",
            },
        ),
        (
            ("--tmp_dir",),
            {"help": "Temporary directory", "default": ".tmp"},
        ),
        (
            ("--is_dicom", "-D"),
            {
                "help": "Assumes input is DICOM (and also converts to DICOM seg; prediction.dcm in output_dir)",
                "action": "store_true",
            },
        ),
        (
            ("--proba_map", "-p"),
            {
                "help": "Produces a Nifti format probability map (probabilities.nii.gz in output_dir)",
                "action": "store_true",
            },
        ),
        (
            ("--proba_threshold",),
            {
                "help": "Sets probabilities in proba_map lower than proba_threhosld to 0",
                "type": float_or_none,
                "default": None,
                "nargs": "+",
            },
        ),
        (
            ("--min_confidence",),
            {
                "help": "Removes objects whose max prob is smaller than min_confidence",
                "type": float_or_none,
                "default": None,
                "nargs": "+",
            },
        ),
        (
            ("--rt_struct_output",),
            {
                "help": "Produces a DICOM RT Struct file (struct.dcm in output_dir; requires DICOM input)",
                "action": "store_true",
            },
        ),
        (
            ("--save_nifti_inputs", "-S"),
            {
                "help": "Moves Nifti inputs to output folder (volume_XXXX.nii.gz in output_dir)",
                "action": "store_true",
            },
        ),
        (
            ("--cascade_mode",),
            {
                "help": "Defines the cascade mode. Must be either intersect or crop.",
                "default": "intersect",
                "type": str,
                "nargs": "+",
                "choices": ["intersect", "crop"],
            },
        ),
        (
            ("--intersect_with",),
            {
                "help": "Calculates the IoU with the SITK mask image in this path and uses this value to filter images such that IoU < --min_intersection are ruled out.",
                "default": None,
                "type": str,
            },
        ),
        (
            ("--min_intersection",),
            {
                "help": "Minimum intersection over the union to keep a candidate.",
                "type": float_or_none,
                "default": 0.1,
                "nargs": "+",
            },
        ),
        (
            ("--crop_from",),
            {
                "help": "Crops the input to the bounding box of the SITK mask image in this path.",
                "default": None,
                "type": str,
            },
        ),
        (
            ("--crop_padding",),
            {
                "help": "Padding to be added to the cropped region.",
                "default": (10, 10, 10),
                "type": int_or_float,
                "nargs": "+",
            },
        ),
        (
            ("--class_idx",),
            {
                "help": "Class index.",
                "default": "all",
                "type": int_or_list_of_ints,
                "nargs": "+",
            },
        ),
        (
            ("--suffix",),
            {
                "help": "Adds a suffix (_suffix) to the outputs if specified.",
                "default": None,
                "type": str,
            },
        ),
    ]
    for arg in args:
        if arg[0][0] in exclude:
            continue
        parser.add_argument(*arg[0], **arg[1])
    return parser
