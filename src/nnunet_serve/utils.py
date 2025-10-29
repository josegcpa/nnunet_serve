"""
General utilities.
"""

import argparse
from glob import glob

import numpy as np
import SimpleITK as sitk
from pydicom import dcmread
from scipy import ndimage

from nnunet_serve.logging_utils import get_logger

logger = get_logger(__name__)

from typing import Sequence

Folds = (
    tuple[int]
    | tuple[int, int]
    | tuple[int, int, int]
    | tuple[int, int, int, int]
    | tuple[int, int, int, int, int]
)

RESCALE_INTERCEPT_TAG = (0x0028, 0x1052)
RESCALE_SLOPE_TAG = (0x0028, 0x1053)


def float_or_none(x: str) -> float | None:
    if x.lower() == "none":
        return None
    return float(x)


def int_or_list_of_ints(x: str) -> int | list[int]:
    x = x.replace(" ", "").strip(",")
    if x.lower() == "all":
        return None
    if "," in x:
        return list(set([int(y) for y in x.split(",")]))
    return int(x)


def list_of_str(x: str) -> list[str]:
    return x.split(",")


def copy_information_nd(
    target_image: sitk.Image, source_image: sitk.Image
) -> sitk.Image:
    """
    Copies information from a source image to a target image. Unlike the
    standard CopyInformation method in SimpleITK, the source image can have
    fewer axes than the target image as long as the first n axes of each are
    identical (where n is the number of axes in the source image).

    Args:
        target_image (sitk.Image): target image.
        source_image (sitk.Image): source information for metadata.

    Raises:
        Exception: if the source image has more dimensions than the target
            image.

    Returns:
        sitk.Image: target image with metadata copied from source image.
            The metadata information for the additional axes is set to 0 in the
            case of the origin, 1.0 in the case of the spacing and to the
            identity in the case of the direction.
    """
    size_source = source_image.GetSize()
    size_target = target_image.GetSize()
    n_dim_in = len(size_source)
    n_dim_out = len(size_target)
    if n_dim_in == n_dim_out:
        target_image.CopyInformation(source_image)
        return target_image
    elif n_dim_in > n_dim_out:
        raise Exception(
            "target_image has to have the same or more dimensions than\
                source_image"
        )
    if size_target[:n_dim_in] != size_source:
        out_str = f"sizes are different (target={size_target[:n_dim_in]}"
        out_str += f" size_source={size_source})"
        return out_str
    spacing = list(source_image.GetSpacing())
    origin = list(source_image.GetOrigin())
    direction = list(source_image.GetDirection())
    while len(origin) != n_dim_out:
        spacing.append(1.0)
        origin.append(0.0)
    direction = np.reshape(direction, (n_dim_in, n_dim_in))
    direction = np.pad(
        direction, ((0, n_dim_out - n_dim_in), (0, n_dim_out - n_dim_in))
    )
    x, y = np.diag_indices(n_dim_out - n_dim_in)
    x = x + n_dim_in
    y = y + n_dim_in
    direction[(x, y)] = 1.0
    target_image.SetSpacing(spacing)
    target_image.SetOrigin(origin)
    target_image.SetDirection(direction.flatten())
    return target_image


def filter_by_bvalue(
    dicom_files: list, target_bvalue: int, exact: bool = False
) -> list:
    """
    Selects the DICOM values with a b-value which is exactly or closest to
    target_bvalue (depending on whether exact is True or False).

    Args:
        dicom_files (list): list of pydicom file objects.
        target_bvalues (int): the expected b-value.
        exact (bool, optional): whether the b-value matching is to be exact
            (raises error if exact target_bvalue is not available) or
            approximate returns the b-value which is closest to target_bvalue.

    Returns:
        list: list of b-value-filtered pydicom file objects.
    """
    BVALUE_TAG = ("0018", "9087")
    SIEMENS_BVALUE_TAG = ("0019", "100c")
    GE_BVALUE_TAG = ("0043", "1039")
    bvalues = []
    for d in dicom_files:
        curr_bvalue = None
        bvalue = d.get(BVALUE_TAG, None)
        siemens_bvalue = d.get(SIEMENS_BVALUE_TAG, None)
        ge_bvalue = d.get(GE_BVALUE_TAG, None)
        if bvalue is not None:
            curr_bvalue = bvalue.value
        elif siemens_bvalue is not None:
            curr_bvalue = siemens_bvalue.value
        elif ge_bvalue is not None:
            curr_bvalue = ge_bvalue.value
            if isinstance(curr_bvalue, bytes):
                curr_bvalue = curr_bvalue.decode()
            curr_bvalue = str(curr_bvalue)
            if "[" in curr_bvalue and "]" in curr_bvalue:
                curr_bvalue = (
                    curr_bvalue.strip().strip("[").strip("]").split(",")
                )
                curr_bvalue = [int(x) for x in curr_bvalue]
            if isinstance(curr_bvalue, list) is False:
                curr_bvalue = curr_bvalue.split("\\")
                curr_bvalue = str(curr_bvalue[0])
            else:
                curr_bvalue = str(curr_bvalue[0])
            if len(curr_bvalue) > 5:
                curr_bvalue = curr_bvalue[-4:]
        if curr_bvalue is None:
            curr_bvalue = 0
        bvalues.append(int(curr_bvalue))
    unique_bvalues = set(bvalues)
    if len(unique_bvalues) in [0, 1]:
        return dicom_files
    if (target_bvalue not in unique_bvalues) and (exact is True):
        raise RuntimeError("Requested b-value not available")
    best_bvalue = sorted(unique_bvalues, key=lambda b: abs(b - target_bvalue))[
        0
    ]
    dicom_files = [f for f, b in zip(dicom_files, bvalues) if b == best_bvalue]
    return dicom_files


def resample_image_to_target(
    moving: sitk.Image,
    target: sitk.Image,
    is_mask: bool = False,
) -> sitk.Image:
    """
    Resamples a SimpleITK image to the space of a target image.

    Args:
      moving: The SimpleITK image to resample.
      target: The target SimpleITK image to match.
      is_mask (bool): whether the moving image is a label mask.

    Returns:
      The resampled SimpleITK image matching the target properties.
    """
    if is_mask:
        interpolation = sitk.sitkLabelLinear
    else:
        interpolation = sitk.sitkBSpline

    output = sitk.Resample(moving, target, sitk.Transform(), interpolation, 0)
    return output


def resample_image(
    sitk_image: sitk.Image,
    out_spacing: Sequence[float] = [1.0, 1.0, 1.0],
    out_size: Sequence[int] = None,
    out_direction: Sequence[float] = None,
    out_origin: Sequence[float] = None,
    is_mask: bool = False,
    interpolator=sitk.sitkLinear,
) -> sitk.Image:
    """Resamples an SITK image to out_spacing. If is_mask is True, uses
    nearest neighbour interpolation. Otherwise, it uses B-splines.

    Args:
        sitk_image (sitk.Image): SITK image.
        out_spacing (Sequence, optional): target spacing for the image.
            Defaults to [1.0, 1.0, 1.0].
        is_mask (bool, optional): sets the interpolation to nearest neighbour.
            Defaults to False.
        interpolator (optional): interpolation method.
            Defaults to sitk.sitkLinear.

    Returns:
        sitk.Image: resampled SITK image.
    """
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()

    if out_direction is None:
        out_direction = sitk_image.GetDirection()

    if out_origin is None:
        out_origin = sitk_image.GetOrigin()

    if out_size is None:
        out_size = [
            round(or_size * (or_spac / out_spac))
            for or_size, or_spac, out_spac in zip(
                original_size, original_spacing, out_spacing
            )
        ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(out_direction)
    resample.SetOutputOrigin(out_origin)
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)

    if is_mask is True:
        resample.SetInterpolator(sitk.sitkLabelLinear)
    else:
        resample.SetInterpolator(interpolator)

    output: sitk.Image = resample.Execute(sitk_image)

    return output


def resample_sitk_bicubic(image, new_spacing):
    old_spacing = np.array(image.GetSpacing())
    old_size = np.array(image.GetSize())

    new_size = np.round(
        old_size * (old_spacing / np.array(new_spacing))
    ).astype(int)

    arr = sitk.GetArrayFromImage(image)

    zoom_factors = old_spacing[::-1] / np.array(new_spacing[::-1])

    arr_resampled = ndimage.zoom(arr, zoom_factors, order=3)

    resampled = sitk.GetImageFromArray(arr_resampled)
    resampled.SetOrigin(image.GetOrigin())
    resampled.SetDirection(image.GetDirection())
    resampled.SetSpacing(new_spacing)

    return resampled


def remove_small_objects(
    image: np.ndarray, min_size: float | int = 0.99
) -> np.ndarray:
    """
    Removes small objects from a multi-label image.

    Args:
        image (np.ndarray): Input multi-label image.
        min_size (float | int, optional): Minimum size of objects to keep in
            voxels. If it is a float, computes the minimum size as a percentage
            of the maximum object size. Defaults to 0.99.

    Returns:
        np.ndarray: Image with small objects removed.
    """
    unique_labels = np.unique(image)
    for u in unique_labels:
        if u == 0:
            continue
        labels, num_features = ndimage.label(image == u)
        label_sizes = {
            i: np.sum(labels == i) for i in range(1, num_features + 1)
        }
        if isinstance(min_size, float):
            curr_min_size = int(min_size * max(label_sizes.values()))
        else:
            curr_min_size = min_size
        for i in range(1, num_features + 1):
            if label_sizes[i] < curr_min_size:
                image[labels == i] = 0
    return image


def mode(a: np.ndarray) -> int | float:
    """
    Calculates the mode of an array.

    Args:
        a (np.ndarray): a numpy array.

    Returns:
        int | float: the mode of a.
    """
    u, c = np.unique(a, return_counts=True)
    return u[np.argmax(c)]


def get_origin(positions: np.ndarray, z_axis: int = 2) -> np.ndarray:
    """
    Returns the origin position from an array of positions (minimum for a given
    z-axis).

    Args:
        positions (np.ndarray): array containing all the positions in a given
            set of arrays.
        z_axis (int, optional): index corresponding to the z-axis. Defaults to
            2.

    Returns:
        np.ndarray: origin of the array.
    """
    origin = positions[positions[:, z_axis].argmin()]
    return origin


def dicom_orientation_to_sitk_direction(
    orientation: Sequence[float],
) -> np.ndarray:
    """Converts the DICOM orientation to SITK orientation. Based on the
    nibabel code that does the same. DICOM uses a more economic encoding
    as one only needs to specify two of the three cosine directions as they
    are all orthogonal. SITK does the more verbose job of specifying all three
    components of the orientation.

    This is based on the Nibabel documentation.

    Args:
        orientation (Sequence[float]): DICOM orientation.

    Returns:
        np.ndarray: SITK (flattened) orientation.
    """
    orientation_array = np.array(orientation).reshape(2, 3).T
    R = np.eye(3)
    R[:, :2] = np.fliplr(orientation_array)
    R[:, 2] = np.cross(orientation_array[:, 1], orientation_array[:, 0])
    R_sitk = np.stack([R[:, 1], R[:, 0], -R[:, 2]], 1)
    return R_sitk.flatten().tolist()


def get_contiguous_arr_idxs(
    positions: np.ndarray, ranking: np.ndarray
) -> np.ndarray | None:
    """
    Uses the ranking to find breaks in positions and returns the elements in
    L which belong to the first contiguous array. Assumes that positions is an
    array of positions (a few of which may be overlapping), ranking is the order
    by which each slice was acquired and d is a dict whose keys will be filtered
    according to this.

    Args:
        positions (np.ndarray): positions with shape [N,3].
        ranking (np.ndarray): ranking used to sort slices.

    Returns:
        np.ndarray: an index vector with the instance numbers of the slices to be kept.
    """
    if all([len(x) == 3 for x in positions]) is False:
        return None
    assert len(positions) == len(ranking)
    order = np.argsort(ranking)
    positions = positions[:, 2][order]
    if len(positions) == 1:
        return None
    p_diff = np.abs(np.round(np.diff(positions) / np.diff(ranking[order]), 1))
    m = mode(p_diff)
    break_points = np.where(np.logical_and(p_diff > 4.5, p_diff > m))[0] + 1
    if len(break_points) == 0:
        return ranking
    segments = np.zeros_like(positions)
    segments[break_points] = 1
    segments = segments.cumsum().astype(int)
    S, C = np.unique(segments, return_counts=True)
    si = S[C >= 8]
    if len(si) == 0:
        return None
    si = si.min()
    output_segment_idxs = ranking[order][segments == si]
    return output_segment_idxs


def read_dicom_as_sitk(file_path: list[str], metadata: dict[str, str] = {}):
    reader = sitk.ImageSeriesReader()
    dicom_file_names = reader.GetGDCMSeriesFileNames(file_path)
    fs = []
    good_file_paths = []
    for dcm_file in dicom_file_names:
        f = dcmread(dcm_file)
        if (0x0020, 0x0037) in f:
            orientation = f[0x0020, 0x0037].value
        if (0x0020, 0x0032) in f:
            fs.append(f)
            good_file_paths.append(dcm_file)
    reader.SetFileNames(good_file_paths)

    sitk_image: sitk.Image = reader.Execute()

    for k in metadata:
        sitk_image.SetMetaData(k, metadata[k])
    return sitk_image, good_file_paths


def get_study_uid(dicom_dir: str) -> str:
    """
    Returns the study UID field from a random file in dicom_dir.

    Args:
        dicom_dir (str): directory with dicom (.dcm) files.

    Returns:
        str: string corresponding to study UID.
    """
    dcm_files = glob(f"{dicom_dir}/*dcm")
    if len(dcm_files) == 0:
        raise RuntimeError(f"No dcm files in {dicom_dir}")
    return dcmread(dcm_files[0])[(0x0020, 0x000D)].value


def export_to_dicom_seg_dcmqi(
    mask_path: str,
    metadata_path: str,
    file_paths: Sequence[Sequence[str]],
    output_dir: str,
    output_file_name: str = "prediction",
) -> str:
    """
    Exports a SITK image mask as a DICOM segmentation object with dcmqi.

    Args:
        mask (sitk.Image): an SITK file object corresponding to a mask.
        mask_path (str): path to (S)ITK mask.
        metadata_path (str): path to metadata template file.
        file_paths (Sequence[str]): list of DICOM file paths corresponding to the
            original series.
        output_dir (str): path to output directory.
        output_file_name (str, optional): output file name. Defaults to
            "prediction".

    Returns:
        str: "success" if the process was successful, "empty mask" if the SITK
            mask contained no values.
    """
    import subprocess

    output_dcm_path = f"{output_dir}/{output_file_name}.dcm"
    logger.info(f"converting to dicom-seg in {output_dcm_path}")
    subprocess.call(
        [
            "itkimage2segimage",
            "--inputDICOMList",
            ",".join(file_paths[0]),
            "--outputDICOM",
            output_dcm_path,
            "--inputImageList",
            mask_path,
            "--inputMetadata",
            metadata_path,
        ]
    )
    return "success"


def export_dicom_files(
    output_dir: str,
    prediction_name: str,
    probabilities_name: str,
    metadata_path: str,
    fractional_metadata_path: str,
    fractional_as_segments: bool,
    dicom_file_paths: list[str],
    proba_map: sitk.Image,
    save_proba_map: bool,
    class_idx: int | None = None,
):
    """
    Convenience function to export DICOM files.

    Args:
        output_dir (str): output directory.
        prediction_name (str): name for binary or multi-class prediction file.
        probabilities_name (str): name for probability prediction file.
        struct_name (str): name for RT struct file.
        metadata_path (str): path to metadata file (required for DICOM seg).
        fractional_metadata_path (str): path to fractional metadata file (
            required for DICOM seg). If ``None`` defaults to ``metadata_path``.
        fractional_as_segments (bool): whether ``proba_map`` should be divided
            into discrete classes according to the number of classes specified
            in ``fractional_metadata_path``.
        dicom_file_paths (list[str]): paths to the input DICOM file paths.
        mask (sitk.Image): image corresponding to mask prediction.
        proba_map (sitk.Image): image corresponding to probability map.
        save_proba_map (bool): whether the probability map should be saved.
        save_rt_struct (bool): whether the RT struct file should be saved.
        class_idx (int | None, optional): index corresponding to the class which
            should be used to export the fractional DICOM seg and the
            exported probability map. Defaults to None.
    """
    mask_path = f"{output_dir}/{prediction_name}.nii.gz"
    export_to_dicom_seg_dcmqi(
        mask_path=mask_path,
        metadata_path=metadata_path,
        file_paths=dicom_file_paths,
        output_dir=output_dir,
        output_file_name=prediction_name,
    )

    if save_proba_map is True and class_idx is not None:
        if fractional_metadata_path is None:
            curr_metadata_path = metadata_path
        else:
            curr_metadata_path = fractional_metadata_path
        export_fractional_dicom_seg(
            proba_map,
            metadata_path=curr_metadata_path,
            file_paths=dicom_file_paths,
            output_dir=output_dir,
            output_file_name=probabilities_name,
            fractional_as_segments=fractional_as_segments,
        )


def calculate_iou(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculates the intersection of the union between arrays a and b.

    Args:
        a (np.ndarray): array.
        b (np.ndarray): array.

    Returns:
        float: float value for the intersection over the union.
    """
    intersection = np.logical_and(a == 1, a == b).sum()
    union = a.sum() + b.sum() - intersection
    return intersection / union


def calculate_iou_a_over_b(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculates how much of a overlaps with b.

    Args:
        a (np.ndarray): array.
        b (np.ndarray): array.

    Returns:
        float: float value for the intersection over the union.
    """
    intersection = np.logical_and(a == 1, a == b).sum()
    union = a.sum()
    return intersection / union


def intersect(
    mask: sitk.Image | np.ndarray,
    intersect_with: sitk.Image | np.ndarray,
    min_intersection: float = 0.1,
) -> sitk.Image | np.ndarray:
    mask_is_sitk = isinstance(mask, sitk.Image)
    if mask_is_sitk:
        mask_arr = sitk.GetArrayFromImage(mask)
    else:
        mask_arr = np.asarray(mask)
    if isinstance(intersect_with, str):
        intersect_with = sitk.ReadImage(intersect_with)
    if isinstance(intersect_with, sitk.Image):
        ref_arr = sitk.GetArrayFromImage(intersect_with)
    else:
        ref_arr = np.asarray(intersect_with)
    ref_arr = (ref_arr > 0).astype(np.uint8)
    binary_mask = (mask_arr > 0).astype(np.uint8)
    labels, num = ndimage.label(binary_mask, structure=np.ones((3, 3, 3)))
    for idx in range(1, num + 1):
        comp = (labels == idx).astype(np.uint8)
        overlap = calculate_iou_a_over_b(comp, ref_arr)
        if overlap < min_intersection:
            mask_arr[comp.astype(bool)] = 0
    if mask_is_sitk:
        out = sitk.GetImageFromArray(mask_arr)
        out.CopyInformation(mask)
        return out
    else:
        return mask_arr


def extract_lesion_candidates(
    softmax: np.ndarray,
    threshold: float = 0.10,
    min_confidence: float = None,
    min_voxels_detection: int = 10,
    max_prob_round_decimals: int = 4,
    intersect_with: str | np.ndarray | sitk.Image = None,
    min_intersection: float = 0.1,
) -> tuple[np.ndarray, list[tuple[int, float]], np.ndarray]:
    """
    Lesion candidate protocol as implemented in [1]. Essentially:

        1. Clips probabilities to be above a threshold
        2. Detects connected components
        3. Filters based on candidate size
        4. Filters based on maximum probability value
        5. Returns the connected components

    [1] https://github.com/DIAGNijmegen/Report-Guided-Annotation/blob/9eef43d3a8fb0d0cb3cfca3f51fda91daa94f988/src/report_guided_annotation/extract_lesion_candidates.py#L17

    Args:
        softmax (np.ndarray): array with softmax probability values.
        threshold (float, optional): threshold below which values are set to 0.
            Defaults to 0.10.
        min_confidence (float, optional): minimum maximum probability value for
            each object after connected component analysis. Defaults to None
            (no filtering).
        min_voxels_detection (int, optional): minimum object size in voxels.
            Defaults to 10.
        max_prob_round_decimals (int, optional): maximum number of decimal
            places. Defaults to 4.
        intersect_with (str | sitk.Image, optional): calculates the
            intersection of each candidate with the image specified in
            intersect_with. If the intersection is larger than
            min_intersection, the candidate is kept; otherwise it is discarded.
            Defaults to None.
        min_intersection (float, optional): minimum intersection over the union to keep
            candidate. Defaults to 0.1.

    Returns:
        tuple[np.ndarray, list[tuple[int, float]], np.ndarray]: the output
            probability map, a list of confidence values, and the connected
            components array as returned by ndimage.label.
    """
    all_hard_blobs = np.zeros_like(softmax)
    confidences = []
    clipped_softmax = softmax.copy()
    clipped_softmax[softmax < threshold] = 0
    blobs_index, num_blobs = ndimage.label(
        clipped_softmax, structure=np.ones((3, 3, 3))
    )
    if min_confidence is None:
        min_confidence = threshold

    if intersect_with is not None:
        logger.info(f"Intersecting with {intersect_with}")
        if isinstance(intersect_with, str):
            intersect_with = sitk.ReadImage(intersect_with)
        if isinstance(intersect_with, sitk.Image):
            intersect_with = sitk.GetArrayFromImage(intersect_with)

    for idx in range(1, num_blobs + 1):
        hard_mask = np.zeros_like(blobs_index)
        hard_mask[blobs_index == idx] = 1

        hard_blob = hard_mask * clipped_softmax
        max_prob = np.max(hard_blob)

        if np.count_nonzero(hard_mask) <= min_voxels_detection:
            blobs_index[hard_mask.astype(bool)] = 0
            continue

        elif max_prob < min_confidence:
            blobs_index[hard_mask.astype(bool)] = 0
            continue

        if intersect_with is not None:
            iou = calculate_iou_a_over_b(hard_mask, intersect_with)
            if iou < min_intersection:
                blobs_index[hard_mask.astype(bool)] = 0
                continue

        if max_prob_round_decimals is not None:
            max_prob = np.round(max_prob, max_prob_round_decimals)
        hard_blob[hard_blob > 0] = clipped_softmax[hard_blob > 0]  # max_prob
        all_hard_blobs += hard_blob
        confidences.append((idx, max_prob))
    return all_hard_blobs, confidences, blobs_index


def make_parser(
    description: str = "Entrypoint for nnUNet prediction. Handles all data "
    "format conversions and cascades of predictions.",
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
    parser.add_argument(
        "--study_path",
        "-i",
        help="Path to input series",
        required=True,
    )
    parser.add_argument(
        "--series_folders",
        "-s",
        nargs="+",
        type=list_of_str,
        help="Path to input series folders",
        required=True,
    )
    parser.add_argument(
        "--nnunet_id",
        nargs="+",
        help="nnUNet ID",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_name",
        help="Checkpoint name for nnUNet",
        default="checkpoint_final.pth",
        nargs="+",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        help="Path to output directory",
        required=True,
    )
    parser.add_argument(
        "--folds",
        "-f",
        help="Sets which folds should be used with nnUNet",
        nargs="+",
        type=int_or_list_of_ints,
        default=(0,),
    )
    parser.add_argument(
        "--tta",
        "-t",
        help="Uses test-time augmentation during prediction",
        action="store_true",
    )
    parser.add_argument(
        "--tmp_dir",
        help="Temporary directory",
        default=".tmp",
    )
    parser.add_argument(
        "--is_dicom",
        "-D",
        help="Assumes input is DICOM (and also converts to DICOM seg; \
            prediction.dcm in output_dir)",
        action="store_true",
    )
    parser.add_argument(
        "--proba_map",
        "-p",
        help="Produces a Nifti format probability map (probabilities.nii.gz \
            in output_dir)",
        action="store_true",
    )
    parser.add_argument(
        "--proba_threshold",
        help="Sets probabilities in proba_map lower than proba_threhosld to 0",
        type=float_or_none,
        default=0.5,
        nargs="+",
    )
    parser.add_argument(
        "--min_confidence",
        help="Removes objects whose max prob is smaller than min_confidence",
        type=float_or_none,
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--rt_struct_output",
        help="Produces a DICOM RT Struct file (struct.dcm in output_dir; requires DICOM input)",
        action="store_true",
    )
    parser.add_argument(
        "--save_nifti_inputs",
        "-S",
        help="Moves Nifti inputs to output folder (volume_XXXX.nii.gz in \
            output_dir)",
        action="store_true",
    )
    parser.add_argument(
        "--cascade_mode",
        help="Defines the cascade mode. Must be either intersect or crop.",
        default="intersect",
        type=str,
        nargs="+",
        choices=["intersect", "crop"],
    )
    parser.add_argument(
        "--intersect_with",
        help="Calculates the IoU with the SITK mask image in this path and uses\
            this value to filter images such that IoU < --min_intersection are ruled out.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--min_intersection",
        help="Minimum intersection over the union to keep a candidate.",
        type=float_or_none,
        default=0.1,
        nargs="+",
    )
    parser.add_argument(
        "--crop_from",
        help="Crops the input to the bounding box of the SITK mask image in this path.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--crop_padding",
        help="Padding to be added to the cropped region.",
        default=(10, 10, 10),
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "--class_idx",
        help="Class index.",
        default="all",
        type=int_or_list_of_ints,
        nargs="+",
    )
    parser.add_argument(
        "--suffix",
        help="Adds a suffix (_suffix) to the outputs if specified.",
        default=None,
        type=str,
    )
    return parser
