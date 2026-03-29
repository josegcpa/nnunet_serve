"""
SimpleITK-based image utilities used during preprocessing and postprocessing.

The functions in this module handle common 3D medical image operations such as
resampling images to a target geometry, changing spacing, and applying
connected-component and morphology operations to masks.
"""

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from typing import Sequence

from nnunetv2.preprocessing.resampling.default_resampling import (
    get_do_separate_z,
)
from nnunet_serve.logging_utils import get_logger

logger = get_logger(__name__)

BASE_INTERPOLATOR = sitk.sitkBSpline
BASE_INTERPOLATOR_Z_AXIS = sitk.sitkNearestNeighbor


def _resample_image_standard(
    sitk_image: sitk.Image,
    out_spacing: Sequence[float] = [1.0, 1.0, 1.0],
    out_size: Sequence[int] = None,
    out_direction: Sequence[float] = None,
    out_origin: Sequence[float] = None,
    is_mask: bool = False,
    interpolator=BASE_INTERPOLATOR,
) -> sitk.Image:
    """Standard resampling of an SITK image.

    Args:
        sitk_image (sitk.Image): SITK image.
        out_spacing (Sequence, optional): target spacing for the image.
            Defaults to [1.0, 1.0, 1.0].
        out_size (Sequence, optional): target size.
        out_direction (Sequence, optional): target direction.
        out_origin (Sequence, optional): target origin.
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


def resample_image_separate_z(
    sitk_image: sitk.Image,
    out_spacing: Sequence[float] = [1.0, 1.0, 1.0],
    is_mask: bool = False,
    interpolator=BASE_INTERPOLATOR,
) -> sitk.Image:
    """Resamples an SITK image to out_spacing with decoupled z and xy axes.

    Args:
        sitk_image (sitk.Image): SITK image.
        out_spacing (Sequence, optional): target spacing for the image.
            Defaults to [1.0, 1.0, 1.0].
        is_mask (bool, optional): sets the interpolation to nearest neighbour.
            Defaults to False.
        interpolator (optional): interpolation method for xy.
            Defaults to sitk.sitkBSpline.

    Returns:
        sitk.Image: resampled SITK image.
    """
    original_spacing = sitk_image.GetSpacing()

    intermediate_spacing = [out_spacing[0], out_spacing[1], original_spacing[2]]
    xy_resampled = _resample_image_standard(
        sitk_image,
        out_spacing=intermediate_spacing,
        is_mask=is_mask,
        interpolator=interpolator,
    )

    z_interpolator = (
        sitk.sitkLabelLinear if is_mask else BASE_INTERPOLATOR_Z_AXIS
    )
    final_resampled = _resample_image_standard(
        xy_resampled,
        out_spacing=out_spacing,
        is_mask=is_mask,
        interpolator=z_interpolator,
    )

    return final_resampled


def resample_image(
    sitk_image: sitk.Image,
    out_spacing: Sequence[float] = [1.0, 1.0, 1.0],
    is_mask: bool = False,
    interpolator=BASE_INTERPOLATOR,
    do_separate_z: bool = False,
    *args,
    **kwargs,
) -> sitk.Image:
    """Resamples an SITK image to out_spacing. Decouples z and xy if required.

    Args:
        sitk_image (sitk.Image): SITK image.
        out_spacing (Sequence, optional): target spacing for the image.
            Defaults to [1.0, 1.0, 1.0].
        is_mask (bool, optional): sets the interpolation to nearest neighbour.
            Defaults to False.
        interpolator (optional): interpolation method.
            Defaults to sitk.sitkBSpline.
        do_separate_z (bool, optional): whether to decouple z and xy axes.
            Defaults to False.
        *args: additional arguments for the resampling function.
        **kwargs: additional keyword arguments for resampling function.

    Returns:
        sitk.Image: resampled SITK image.
    """
    # SimpleITK spacing is (x, y, z). Reverse for nnUNet which expects (z, y, x).
    original_spacing = sitk_image.GetSpacing()
    do_separate_z = get_do_separate_z(original_spacing[::-1]) and do_separate_z

    if do_separate_z:
        return resample_image_separate_z(
            sitk_image,
            out_spacing=out_spacing,
            is_mask=is_mask,
            interpolator=interpolator,
            *args,
            **kwargs,
        )
    else:
        return _resample_image_standard(
            sitk_image,
            out_spacing=out_spacing,
            is_mask=is_mask,
            interpolator=interpolator,
            *args,
            **kwargs,
        )


def resample_image_to_target(
    moving: sitk.Image,
    target: sitk.Image,
    is_mask: bool = False,
    interpolator=BASE_INTERPOLATOR,
) -> sitk.Image:
    """
    Resamples a SimpleITK image to the space of a target image.

    Args:
        moving: The SimpleITK image to resample.
        target: The target SimpleITK image to match.
        is_mask (bool): whether the moving image is a label mask.
        interpolator (sitk.sitkInterpolation): interpolation method.

    Returns:
        The resampled SimpleITK image matching the target properties.
    """
    target_size = target.GetSize()
    target_spacing = target.GetSpacing()
    target_direction = target.GetDirection()
    target_origin = target.GetOrigin()

    output = _resample_image_standard(
        moving,
        out_spacing=target_spacing,
        out_size=target_size,
        out_direction=target_direction,
        out_origin=target_origin,
        is_mask=is_mask,
        interpolator=interpolator,
    )

    return output


def to_closest_canonical_sitk(
    img: sitk.Image | list[sitk.Image],
) -> sitk.Image | list[sitk.Image]:
    if isinstance(img, list):
        return [to_closest_canonical_sitk(i) for i in img]
    direction = img.GetDirection()
    direction_matrix = np.array(direction).reshape(3, 3)
    flip_axes = [
        i
        for i in range(3)
        if np.dot(direction_matrix[:, i], np.eye(3)[:, i]) < 0
    ]

    canonical_img = img
    for axis in flip_axes:
        canonical_img = sitk.Flip(
            canonical_img, [axis == i for i in range(3)], False
        )

    canonical_img.SetDirection(tuple(np.eye(3).flatten()))
    return canonical_img


def from_closest_canonical_sitk(
    canonical_img: sitk.Image | list[sitk.Image],
    original_img: sitk.Image | list[sitk.Image],
) -> sitk.Image | list[sitk.Image]:
    if isinstance(canonical_img, list):
        return [
            from_closest_canonical_sitk(i, o)
            for i, o in zip(canonical_img, original_img)
        ]
    if isinstance(original_img, list):
        original_ref = original_img[0]
    else:
        original_ref = original_img

    original_direction = np.array(original_ref.GetDirection()).reshape(3, 3)
    canonical_direction = np.eye(3)

    flip_axes = [
        i
        for i in range(3)
        if np.dot(original_direction[:, i], canonical_direction[:, i]) < 0
    ]

    restored_img = canonical_img
    for axis in flip_axes:
        restored_img = sitk.Flip(
            restored_img, [axis == i for i in range(3)], False
        )

    restored_img.CopyInformation(original_ref)
    return restored_img


def get_crop(
    image: str | sitk.Image,
    target_image: sitk.Image | None = None,
    crop_padding: tuple[int | float, int | float, int | float]
    | None = (
        10,
        10,
        10,
    ),
    min_size: tuple[int, int, int] | None = None,
) -> tuple[int, int, int, int, int, int]:
    """
    Retrieves the bounding box of a label in an image.

    Args:
        image (str | sitk.Image): input image.
        target_image (sitk.Image | None, optional): target image. Defaults to None.
        crop_padding (tuple[int, int, int] | None, optional): padding to be added to
            the cropped region. Defaults to None.
        min_size (tuple[int, int, int] | None, optional): minimum size of the cropped
            region. Defaults to None.

    Returns:
        tuple[int, int, int, int, int, int]: bounding box of the label.
    """
    logger.info("Cropping input")
    logger.info("Input size (before cropping): %s", image.GetSize())
    if isinstance(image, str):
        image = sitk.ReadImage(image)
    if target_image is not None:
        image = resample_image_to_target(
            image, target=target_image, is_mask=True
        )
    target_image_size = image.GetSize()
    labelimfilter = sitk.LabelShapeStatisticsImageFilter()
    labelimfilter.Execute(image)
    bounding_box = labelimfilter.GetBoundingBox(1)
    start, size = bounding_box[:3], bounding_box[3:]

    if crop_padding is not None:
        crop_padding = [
            c if isinstance(c, int) else int(c * s)
            for c, s in zip(crop_padding, size)
        ]
        logger.info("Crop padding: %s", crop_padding)
        start = tuple([max(start[i] - crop_padding[i], 0) for i in range(3)])
        size = tuple(
            [
                min(
                    int(size[i] + crop_padding[i] * 2),
                    int(target_image_size[i] - start[i]),
                )
                for i in range(3)
            ]
        )
    if min_size is not None:
        for i in range(3):
            if size[i] < min_size[i]:
                new_start = max(start[i] - (min_size[i] - size[i]) // 2, 0)
                start[i] = new_start
                size[i] = min_size[i]

    bounding_box = [
        start[0],
        start[1],
        start[2],
        start[0] + size[0],
        start[1] + size[1],
        start[2] + size[2],
    ]
    output_padding = [
        bounding_box[0],
        bounding_box[1],
        bounding_box[2],
        target_image_size[0] - bounding_box[3],
        target_image_size[1] - bounding_box[4],
        target_image_size[2] - bounding_box[5],
    ]
    output_padding = list(map(int, output_padding))

    logger.info("Output bounding box corner (top): %s", bounding_box[:3])
    logger.info("Output bounding box corner (bottom): %s", bounding_box[3:])
    logger.info("Output bounding box size: %s", size)
    logger.info("Output padding: %s", output_padding)

    return bounding_box, output_padding
