import json
import os
import subprocess as sp
from enum import Enum
from glob import glob
from typing import Any, Union

import numpy as np
import SimpleITK as sitk
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.dataloading.multi_threaded_augmenter import (
    MultiThreadedAugmenter,
)
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)
from pydantic import BaseModel, ConfigDict, Field

from nnunet_serve.logging_utils import get_logger
from nnunet_serve.seg_writers import SegWriter
from nnunet_serve.utils import (
    Folds,
    copy_information_nd,
    extract_lesion_candidates,
    intersect,
    read_dicom_as_sitk,
    resample_image_to_target,
)

logger = get_logger(__name__)
SUCCESS_STATUS = "done"
FAILURE_STATUS = "failed"

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa


class CascadeMode(Enum):
    INTERSECT = "intersect"
    CROP = "crop"


class InferenceRequest(BaseModel):
    """
    Data model for the inference request from local data. Supports providing
    multiple nnUNet model identifiers (``nnunet_id``) which in turn allows for
    intersection-based filtering of downstream results.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    nnunet_id: str | list[str] = Field(
        description="nnUnet model identifier or list of nnUNet model identifiers."
    )
    study_path: str = Field(
        description="Path to study folder or list of paths to studies."
    )
    series_folders: list[str] | list[list[str]] = Field(
        description="Series folder names or list of series folder names (relative to study_path).",
        default=None,
    )
    output_dir: str = Field(description="Output directory.")
    class_idx: int | list[int | None] | list[list[int] | int | None] = Field(
        description="Prediction index or indices which are kept after each prediction",
        default=1,
    )
    checkpoint_name: str = Field(
        description="nnUNet checkpoint name", default=None
    )
    tmp_dir: str = Field(
        description="Directory for temporary outputs", default=".tmp"
    )
    is_dicom: bool = Field(
        description="Whether series_paths refers to DICOM series folders",
        default=False,
    )
    tta: bool = Field(
        description="Whether to apply test-time augmentation (use_mirroring)",
        default=True,
    )
    use_folds: Folds = Field(
        description="Which folds should be used", default_factory=lambda: [0]
    )
    proba_threshold: float | list[float | None] | None = Field(
        description="Probability threshold for model output", default=None
    )
    min_confidence: float | list[float | None] | None = Field(
        description="Minimum confidence for model output", default=None
    )
    intersect_with: str | None = Field(
        description="Intersects output with this mask and if relative \
            intersection < min_overlap this is set to 0",
        default=None,
    )
    min_overlap: float = Field(
        description="Minimum overlap for intersection", default=0.1
    )
    crop_from: str | None = Field(
        description="Crops input to the bounding box of this model. ",
        default=None,
    )
    crop_padding: tuple[int, int, int] | None = Field(
        description="Padding to be added to the cropped region", default=None
    )
    cascade_mode: CascadeMode = Field(
        description="Whether to crop inputs to consecutive bounding boxes "
        "or to intersect consecutive outputs.",
        default=CascadeMode.INTERSECT,
    )
    save_proba_map: bool = Field(
        description="Saves the probability map", default=False
    )
    save_nifti_inputs: bool = Field(
        description="Saves the Nifti inputs in the output folder if input is DICOM",
        default=False,
    )
    save_rt_struct_output: bool = Field(
        description="Saves the output as an RT struct file", default=False
    )
    suffix: str | None = Field(
        description="Suffix for predictions", default=None
    )

    def model_post_init(self, context):
        if self.save_proba_map and all(
            [x is None for x in self.proba_threshold]
        ):
            raise ValueError(
                "proba_threshold must be not-None if save_proba_map is True"
            )


def convert_predicted_logits_to_segmentation_with_correct_shape(
    predicted_logits: Union[torch.Tensor, np.ndarray],
    plans_manager: PlansManager,
    configuration_manager: ConfigurationManager,
    label_manager: LabelManager,
    properties_dict: dict,
    return_probabilities: bool = False,
    num_threads_torch: int = default_num_processes,
):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape
    current_spacing = (
        configuration_manager.spacing
        if len(configuration_manager.spacing)
        == len(properties_dict["shape_after_cropping_and_before_resampling"])
        else [properties_dict["spacing"][0], *configuration_manager.spacing]
    )

    predicted_logits = configuration_manager.resampling_fn_probabilities(
        predicted_logits,
        properties_dict["shape_after_cropping_and_before_resampling"],
        current_spacing,
        properties_dict["spacing"],
    )

    # return value of resampling_fn_probabilities can be ndarray or Tensor but that does not matter because
    # apply_inference_nonlin will convert to torch
    predicted_probabilities = label_manager.apply_inference_nonlin(
        predicted_logits
    )
    del predicted_logits
    segmentation = label_manager.convert_probabilities_to_segmentation(
        predicted_probabilities
    )

    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(
        properties_dict["shape_before_cropping"],
        dtype=(
            np.uint8
            if len(label_manager.foreground_labels) < 255
            else np.uint16
        ),
    )
    slicer = bounding_box_to_slice(properties_dict["bbox_used_for_cropping"])
    segmentation_reverted_cropping[slicer] = segmentation
    del segmentation
    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(
        plans_manager.transpose_backward
    )
    if return_probabilities:
        # revert cropping
        predicted_probabilities = (
            label_manager.revert_cropping_on_probabilities(
                predicted_probabilities,
                properties_dict["bbox_used_for_cropping"],
                properties_dict["shape_before_cropping"],
            )
        )
        predicted_probabilities = predicted_probabilities.cpu().numpy()
        # revert transpose
        predicted_probabilities = predicted_probabilities.transpose(
            [0] + [i + 1 for i in plans_manager.transpose_backward]
        )
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping, predicted_probabilities
    else:
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping


def predict_from_data_iterator_local(
    predictor: nnUNetPredictor,
    data_iterator: list[dict[str, Any]],
    save_probabilities: bool = False,
    class_idx: int | list[int] | None = None,
):
    """
    Adapts the original predict_from_data_iterator to use no multiprocessing.
    """
    ret = []
    for preprocessed in data_iterator:
        data = preprocessed["data"]
        if isinstance(data, str):
            delfile = data
            data = torch.from_numpy(np.load(data))
            os.remove(delfile)

        properties = preprocessed["data_properties"]

        prediction = predictor.predict_logits_from_preprocessed_data(data).cpu()
        n_classes = prediction.shape[0]
        logger.info("nnUNet: predicted logits")
        if class_idx is not None:
            old_labels = predictor.label_manager._all_labels
            old_regions = predictor.label_manager._regions
            if isinstance(class_idx, int):
                class_idx = [class_idx]
            used_labels = [0, *class_idx]
            prediction = prediction[[0, *class_idx]]
            if predictor.label_manager._has_regions:
                new_regions = [
                    r
                    for r in predictor.label_manager._regions
                    if any([rr in used_labels for rr in r])
                ]
                used_labels = (
                    np.unique(np.concatenate(new_regions)).sort().tolist()
                )
            predictor.label_manager._all_labels = used_labels

        logger.info("nnUNet: resampling...")
        processed_output = (
            convert_predicted_logits_to_segmentation_with_correct_shape(
                prediction,
                predictor.plans_manager,
                predictor.configuration_manager,
                predictor.label_manager,
                properties,
                save_probabilities,
            )
        )
        if class_idx is not None:
            if save_probabilities is False:
                corrected_mask = np.zeros_like(processed_output)
                for i, label in enumerate(used_labels[1:]):
                    corrected_mask = np.where(
                        processed_output[i] == i + 1,
                        label,
                        corrected_mask,
                    )
            else:
                corrected_mask = np.zeros_like(processed_output[0])
                corrected_prob = np.zeros(
                    [n_classes, *processed_output[0].shape]
                )
                for i, label in enumerate(used_labels[1:]):
                    corrected_mask = np.where(
                        processed_output[i] == i + 1,
                        label,
                        corrected_mask,
                    )
                    corrected_prob[label] = processed_output[i]

        logger.info("nnUNet: converted logits to segmentation")
        ret.append(processed_output)
        logger.info(f"Done with image of shape {data.shape}")
        predictor.label_manager._all_labels = old_labels
        predictor.label_manager._regions = old_regions

    if isinstance(data_iterator, MultiThreadedAugmenter):
        data_iterator._finish()

    # clear lru cache
    compute_gaussian.cache_clear()
    # clear device cache
    empty_cache(predictor.device)
    return ret


def get_gpu_memory() -> list[int]:
    """
    Utility to retrieve value for free GPU memory.

    Returns:
        list[int]: list of available GPU memory (each one corresponds to a GPU
            index).
    """
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [
        int(x.split()[0]) for i, x in enumerate(memory_free_info)
    ]
    return memory_free_values


def get_series_paths(
    study_path: str,
    series_folders: list[str] | list[list[str]] | None,
    n: int | None,
) -> tuple[list[str] | list[list[str]], str, str]:
    """
    Gets the complete paths for series given a ``study_path`` and the names of
    ``series_folders``. Given ``n``, which is the number of nnUNet models which
    will be running, this returns different values:

    * When ``n is None``: returns a list of paths, a status message, and a
        possible error message.
    * When ``n is not None and n > 0``: returns a list of list of paths, a
        status message and a possible error message.

    Args:
        study_path (str): path to study.
        series_folders (list[str] | list[list[str]] | None): series folder names
            relative to ``study_path``.
        n (int | None): number of nnUNet models to run. If None assumes a single
            model is run.

    Returns:
        tuple[list[str], str, str] | tuple[list[list[str]], str, str]: _description_
    """
    if series_folders is None:
        return (
            None,
            FAILURE_STATUS,
            "series_folders must be defined",
        )
    if n is None:
        series_paths = [os.path.join(study_path, x) for x in series_folders]
    else:
        study_path = [study_path for _ in range(n)]
        series_paths = []
        if n != len(series_folders):
            return (
                None,
                FAILURE_STATUS,
                "series_folders and nnunet_id must be the same length",
            )
        for i in range(len(study_path)):
            series_paths.append(
                [os.path.join(study_path[i], x) for x in series_folders[i]]
            )

    return series_paths, SUCCESS_STATUS, None


def wait_for_gpu(min_mem: int) -> int:
    """
    Waits for a GPU with at least ``min_mem`` free memory to be free.

    Args:
        min_mem (int): minimum amount of memory.

    Returns:
        int: GPU ID corresponding to freest GPU.
    """
    free = False
    while free is False:
        gpu_memory = get_gpu_memory()
        max_gpu_memory = max(gpu_memory)
        device_id = [
            i for i in range(len(gpu_memory)) if gpu_memory[i] == max_gpu_memory
        ][0]
        if max_gpu_memory > min_mem:
            free = True
    return device_id


def get_default_params(default_args: dict | list[dict]) -> dict:
    """
    Returns a dict with default parameters. If ``default_args`` is a list of
    dicts, the output will be a dictionary of lists whenever the key is in
    ``args_with_mult_support`` and whose value will be that of the last
    dictionary otherwise. If ``default_args`` is a dict the output will be
    ``default_args``.

    Args:
        default_args (dict | list[dict]): default arguments.

    Returns:
        dict: correctly formatted default arguments.
    """
    args_with_mult_support = [
        "proba_threshold",
        "min_confidence",
        "class_idx",
        "series_folders",
    ]
    if isinstance(default_args, dict):
        default_params = default_args
    elif isinstance(default_args, list):
        default_params = {}
        for curr_default_args in default_args:
            for k in curr_default_args:
                if k in args_with_mult_support:
                    if k not in default_params:
                        default_params[k] = []
                    default_params[k].append(curr_default_args[k])
                else:
                    default_params[k] = curr_default_args[k]
    else:
        raise ValueError("default_args should either be dict or list")
    return default_params


def predict(
    series_paths: list,
    metadata: str,
    mirroring: bool,
    device_id: int,
    params: dict,
    nnunet_path: str | list[str],
) -> list[str]:
    """
    Runs the prediction for a set of models.

    Args:
        series_paths (list): paths to series.
        metadata (str): DICOM seg metadata. Has to be a dict with either "path"
            (pointing towards a DCMQI metadata file) or a list of metadata
            key-value pairs (please see ``SegWriter`` for details).
        mirroring (bool): whether to use mirroring during inference.
        device_id (int): GPU identifier.
        params (dict): parameters which will be used in wraper.
        nnunet_path (str | list[str]): path or paths to nnUNet model.

    Returns:
        list[str]: list of output paths.
    """
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=mirroring,
        device=torch.device("cuda", device_id),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    inference_param_names = [
        "output_dir",
        "class_idx",
        "checkpoint_name",
        "tmp_dir",
        "is_dicom",
        "use_folds",
        "proba_threshold",
        "min_confidence",
        "intersect_with",
        "crop_from",
        "crop_padding",
        "min_overlap",
        "cascade_mode",
    ]
    export_param_names = [
        "output_dir",
        "suffix",
        "is_dicom",
        "save_proba_map",
        "save_nifti_inputs",
        "save_rt_struct_output",
    ]
    delete_params = [
        "nnunet_id",
        "tta",
        "min_mem",
        "aliases",
        "study_path",
        "series_folders",
    ]

    params = {k: params[k] for k in params if k not in delete_params}
    inference_params = {
        k: params[k] for k in params if k in inference_param_names
    }
    export_params = {k: params[k] for k in params if k in export_param_names}

    all_predictions, all_proba_maps, good_file_paths, all_volumes = (
        multi_model_inference(
            series_paths=series_paths,
            predictor=predictor,
            nnunet_path=nnunet_path,
            **inference_params,
        )
    )

    output_paths = export_predictions(
        masks=all_predictions,
        proba_maps=all_proba_maps,
        good_file_paths=good_file_paths,
        volumes=all_volumes,
        metadata=metadata,
        **export_params,
    )

    del predictor
    torch.cuda.empty_cache()
    return output_paths


def get_crop(
    image: str | sitk.Image,
    target_image: sitk.Image | None = None,
    crop_padding: tuple[int, int, int] | None = None,
    min_size: tuple[int, int, int] | None = None,
) -> tuple[int, int, int, int, int, int]:
    """
    Retrieves the bounding box of a label in an image.

    Args:
        image (str | sitk.Image): input image.
        target_image (sitk.Image | None, optional): target image. Defaults to None.
        crop_padding (tuple[int, int, int] | None, optional): padding to be added to the cropped region. Defaults to None.
        min_size (tuple[int, int, int] | None, optional): minimum size of the cropped region. Defaults to None.

    Returns:
        tuple[int, int, int, int, int, int]: bounding box of the label.
    """
    class_idx = 1
    if isinstance(image, str):
        if ":" in image:
            image, class_idx = image.split(":")
            class_idx = int(class_idx)
        image = sitk.ReadImage(image)
    if target_image is not None:
        # check if resampling is required
        if any(
            [
                image.GetSpacing() != target_image.GetSpacing(),
                image.GetOrigin() != target_image.GetOrigin(),
                image.GetDirection() != target_image.GetDirection(),
            ]
        ):
            image = resample_image_to_target(
                image, target=target_image, is_label=True
            )
    target_image_size = target_image.GetSize()
    image = image == class_idx
    labelimfilter = sitk.LabelShapeStatisticsImageFilter()
    labelimfilter.Execute(image)
    bounding_box = labelimfilter.GetBoundingBox(class_idx)
    start, size = bounding_box[:3], bounding_box[3:]
    if crop_padding is not None:
        for i in range(3):
            start[i] = max(start[i] - crop_padding[i], 0)
            size[i] = min(size[i] + crop_padding[i], target_image_size[i])
    if min_size is not None:
        for i in range(3):
            if size[i] < min_size[i]:
                new_start = max(
                    bounding_box[i] - (min_size[i] - size[i]) // 2, 0
                )
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
    return bounding_box, output_padding


def load_series(series_paths: list[list[str]], is_dicom: bool = False):
    unique_series_paths = []
    for series_path in series_paths:
        for s in series_path:
            if s not in unique_series_paths:
                unique_series_paths.append(s)
    unique_volumes = {}
    all_good_file_paths = {}
    for s in unique_series_paths:
        if is_dicom:
            unique_volumes[s], all_good_file_paths[s] = read_dicom_as_sitk(
                glob(f"{s}/*dcm")
            )
        else:
            unique_volumes[s] = sitk.ReadImage(s)
    logger.info("Loaded %d unique volumes", len(unique_volumes))
    all_volumes = []
    for i, series_path in enumerate(series_paths):
        if len(series_path) > 1:
            # resample to first using unique_volumes
            curr_volumes = [unique_volumes[s]] + [
                resample_image_to_target(
                    unique_volumes[s], unique_volumes[series_path[0]]
                )
                for s in series_path[1:]
            ]
        else:
            curr_volumes = [unique_volumes[series_path[0]]]
        all_volumes.append(curr_volumes)
    return all_volumes, all_good_file_paths


def process_proba_array(
    proba_array: np.ndarray,
    input_image: sitk.Image,
    proba_threshold: float = 0.1,
    min_confidence: float = 0.5,
    intersect_with: str | sitk.Image | None = None,
    min_intersection: float = 0.1,
    class_idx: int | list[int] | None = None,
    output_padding: list[int] | None = None,
) -> sitk.Image:
    """
    Exports a SITK probability mask and the corresponding prediction. Applies a
    candidate extraction protocol (i.e. filtering probabilities above
    proba_threshold, applying connected component analysis and filtering out
    objects whose maximum probability is lower than min_confidence).

    Args:
        array (np.ndarray): an array corresponding to a probability map.
        proba_threshold (float, optional): sets values below this value to 0.
        min_confidence (float, optional): removes objects whose maximum
            probability is lower than this value.
        intersect_with (str | sitk.Image, optional): calculates the
            intersection of each candidate with the image specified in
            intersect_with. If the intersection is larger than
            min_intersection, the candidate is kept; otherwise it is discarded.
            Defaults to None.
        min_intersection (float, optional): minimum intersection over the union
            to keep candidate. Defaults to 0.1.
        class_idx (int | list[int] | None, optional): class index for output
            probability. Defaults to None (no selection).
        output_padding (list[int] | None, optional): padding to apply to the
            output mask. Defaults to None.

    Returns:
        sitk.Image: returns the probability mask after the candidate extraction
            protocol.
    """
    logger.info("Exporting probability map and mask")
    empty = False
    if class_idx is None:
        mask = np.argmax(proba_array, 0)
        proba_array = np.moveaxis(proba_array, 0, -1)
    else:
        proba_array = np.where(proba_array < proba_threshold, 0.0, proba_array)
        if isinstance(class_idx, int):
            proba_array = proba_array[class_idx]
        elif isinstance(class_idx, (list, tuple)):
            proba_array = proba_array[class_idx].sum(0)
        proba_array, _, _ = extract_lesion_candidates(
            proba_array,
            threshold=proba_threshold,
            min_confidence=min_confidence,
            intersect_with=intersect_with,
            min_intersection=min_intersection,
        )
        mask = proba_array > proba_threshold
        proba_map = sitk.GetImageFromArray(proba_array)
        proba_map = copy_information_nd(proba_map, input_image)
        mask = sitk.GetImageFromArray(mask.astype(np.uint32))
        mask.CopyInformation(input_image)
    mask_stats = sitk.LabelShapeStatisticsImageFilter()
    mask_stats.Execute(mask)
    if len(mask_stats.GetLabels()) == 0:
        logger.warning("Mask is empty")
        empty = True
    if output_padding is not None:
        pad_filter = sitk.ConstantPadImageFilter()
        pad_filter.SetPadLowerBound(output_padding[:3])
        pad_filter.SetPadUpperBound(output_padding[3:])
        mask = pad_filter.Execute(mask)
        proba_map = pad_filter.Execute(proba_map)

    return mask, proba_map, empty


def process_mask(
    mask_array: np.ndarray,
    input_image: sitk.Image,
    intersect_with: str | sitk.Image | None = None,
    min_intersection: float = 0.1,
    output_padding: tuple[int, int, int, int, int, int] | None = None,
) -> sitk.Image:
    """
    Exports a SITK probability mask and the corresponding prediction. Applies a
    candidate extraction protocol (i.e. filtering probabilities above
    proba_threshold, applying connected component analysis and filtering out
    objects whose maximum probability is lower than min_confidence).

    Args:
        mask_array (np.ndarray): an array corresponding to a mask.
        intersect_with (str | sitk.Image, optional): calculates the
            intersection of each candidate with the image specified in
            intersect_with. If the intersection is larger than
            min_intersection, the candidate is kept; otherwise it is discarded.
            Defaults to None.
        min_intersection (float, optional): minimum intersection over the union
            to keep candidate. Defaults to 0.1.
        output_padding (list[int] | None, optional): padding to apply to the
            output mask. Defaults to None.

    Returns:
        sitk.Image: returns the probability mask after the candidate extraction
            protocol.
    """
    logger.info("Exporting mask")
    empty = False
    if intersect_with is not None:
        mask_array = intersect(mask_array, intersect_with, min_intersection)
    mask = sitk.GetImageFromArray(mask_array)
    mask.CopyInformation(input_image)
    mask_stats = sitk.LabelShapeStatisticsImageFilter()
    mask_stats.Execute(mask)
    logger.info("Labels: %s", mask_stats.GetLabels())
    if len(mask_stats.GetLabels()) == 0:
        logger.warning("Mask is empty")
        empty = True
    if output_padding is not None:
        pad_filter = sitk.ConstantPadImageFilter()
        pad_filter.SetPadLowerBound(output_padding[:3])
        pad_filter.SetPadUpperBound(output_padding[3:])
        mask = pad_filter.Execute(mask)
    return mask, empty


def single_model_inference(
    predictor: nnUNetPredictor,
    nnunet_path: str,
    volumes: list[sitk.Image],
    output_dir: str,
    class_idx: int | list[int] = 1,
    checkpoint_name: str = "checkpoint_best.pth",
    tmp_dir: str = ".tmp",
    use_folds: Folds = (0,),
    proba_threshold: float | None = None,
    min_confidence: float | None = None,
    intersect_with: str | sitk.Image | None = None,
    crop_from: str | sitk.Image | None = None,
    crop_padding: tuple[int, int, int] | None = None,
    min_overlap: float = 0.1,
) -> tuple[list[str], str, list[list[str]], sitk.Image]:
    """
    Runs the inference for a single model.

    Args:
        predictor (nnUNetPredictor): nnUNet predictor.
        nnunet_path (str): path to nnUNet model.
        series (list[str]): series volumes or paths to series.
        output_dir (str): output directory.
        class_idx (int | list[int], optional): class index for probability
            output. Defaults to 1.
        checkpoint_name (str, optional): name of checkpoint in nnUNet model.
            Defaults to "checkpoint_best.pth".
        tmp_dir (str, optional): directory where temporary outputs are stored.
            Defaults to ".tmp".
        is_dicom (bool, optional): whether the input/output is DICOM. Defaults
            to False.
        use_folds (Folds, optional): which folds from the nnUNet model will be
            used. Defaults to (0,).
        proba_threshold (float, optional): probability threshold to consider a
            pixel positive positive. Defaults to 0.1.
        min_confidence (float | None, optional): minimum confidence level for
            each detected object. Defaults to None.
        intersect_with (str | sitk.Image | None, optional): whether the
            prediction should intersect with a given object. Defaults to None.
        crop_from (str | sitk.Image | None, optional): whether the
            input should be cropped centered on a given mask object. If
            specified as a string, it can be either the path or the path:class_idx.
            Defaults to None.
        crop_padding (tuple[int, int, int] | None, optional): padding to be
            added to the cropped region. Defaults to None.
        min_overlap (float, optional): fraction of prediction which should
            intersect with ``intersect_with``. Defaults to 0.1.
        prediction_name (str | None, optional): name of the prediction. Defaults
            to None.

    Raises:
        ValueError: if there is a mismatch between the number of series and
            the number of channels in the model.

    Returns:
        tuple[list[str], str, list[list[str]], sitk.Image]: prediction files,
            path to output mask, good DICOM file paths, probability map.
    """

    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        nnunet_path,
        use_folds=use_folds,
        checkpoint_name=checkpoint_name,
    )
    patch_size = predictor.configuration_manager.configuration["patch_size"]

    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    predictor.dataset_json["file_ending"] = ".nii.gz"
    if len(predictor.dataset_json["channel_names"]) != len(volumes):
        exp_chan = predictor.dataset_json["channel_names"]
        raise ValueError(
            f"series_paths should have length {len(exp_chan)} ({exp_chan}) but has length {len(volumes)}"
        )
    output_padding = None
    if crop_from is not None:
        logger.info("Cropping input")
        bb, output_padding = get_crop(
            crop_from, volumes[0], crop_padding, patch_size[::-1]
        )
        volumes = [
            v[bb[0] : bb[3], bb[1] : bb[4], bb[2] : bb[5]] for v in volumes
        ]

    logger.info("Running inference using %s", nnunet_path)
    input_array = np.stack([sitk.GetArrayFromImage(v) for v in volumes])
    image_properties = {
        "spacing": volumes[0].GetSpacing()[::-1],
        "origin": volumes[0].GetOrigin(),
        "direction": volumes[0].GetDirection(),
        "size": volumes[0].GetSize(),
    }
    logger.info("Input shape: %s", input_array.shape)
    logger.info("Input spacing: %s", volumes[0].GetSpacing())
    logger.info("Input origin: %s", volumes[0].GetOrigin())
    logger.info("Input direction: %s", volumes[0].GetDirection())
    logger.info("Input size: %s", volumes[0].GetSize())
    logger.info("nnUNet: creating data iterator")
    iterator = predictor.get_data_iterator_from_raw_npy_data(
        [input_array], None, [image_properties], None, 1
    )
    logger.info("nnUNet: running inference")
    mask_array, proba_array = predict_from_data_iterator_local(
        predictor, iterator, save_probabilities=True, class_idx=class_idx
    )[0]
    logger.info("nnUNet: inference done")

    if proba_threshold is not None:
        mask, probability_map, _ = process_proba_array(
            proba_array,
            volumes[0],
            min_confidence=min_confidence,
            proba_threshold=proba_threshold,
            intersect_with=intersect_with,
            min_intersection=min_overlap,
            class_idx=class_idx,
            output_padding=output_padding,
        )
    else:
        mask, _ = process_mask(
            mask_array,
            volumes[0],
            intersect_with=intersect_with,
            min_intersection=min_overlap,
            output_padding=output_padding,
        )
        probability_map = None

    logger.info("Finished processing masks")

    return mask, probability_map


def get_info(dataset_json_path: str) -> dict:
    """
    Loads an nnUNet dataset JSON path.

    Args:
        dataset_json_path (str): path to dataset JSON.

    Returns:
        dict: the dataset JSON.
    """
    with open(dataset_json_path) as o:
        return json.load(o)


def multi_model_inference(
    predictor: nnUNetPredictor,
    nnunet_path: str | list[str],
    series_paths: list[str] | list[list[str]],
    output_dir: str,
    class_idx: int | list[int] | list[list[int]] = 1,
    checkpoint_name: str = "checkpoint_best.pth",
    tmp_dir: str = ".tmp",
    is_dicom: bool = False,
    use_folds: tuple[int] = (0,),
    proba_threshold: float | tuple[float] | list[float] = 0.1,
    min_confidence: float | tuple[float] | list[float] | None = None,
    intersect_with: str | sitk.Image | None = None,
    min_overlap: float = 0.1,
    crop_from: str | sitk.Image | None = None,
    crop_padding: tuple[int, int, int] | None = None,
    cascade_mode: str = "intersect",
):
    """
    Prediction wraper for multiple models. Exports the outputs.

    Args:
        predictor (nnUNetPredictor): nnUNetPredictor object.
        nnunet_path (str | list[str]): path or paths to nnUNet models.
        series_paths (list[str] | list[list[str]]): list of paths or list of
            list of paths corresponding to series.
        output_dir (str): output directory.
        class_idx (int | list[int] | list[list[int]], optional): class index to
            export probability maps. Defaults to 1.
        checkpoint_name (str, optional): name of nnUNet checkpoint. Defaults to
            "checkpoint_best.pth".
        tmp_dir (str, optional): temporary directory. Defaults to ".tmp".
        is_dicom (bool, optional): whether the input/output is DICOM. Defaults
            to False.
        use_folds (tuple[int], optional): which folds should be used. Defaults
            to (0,).
        proba_threshold (float | tuple[float] | list[float], optional):
            probability threshold to consider a pixel positive. Defaults to 0.1.
        min_confidence (float | tuple[float] | list[float] | None, optional):
            minimum confidence to keep an object. Defaults to None.
        intersect_with (str | sitk.Image | None, optional): whether the
            prediction should intersect with a given object. Defaults to None.
        min_overlap (float, optional): fraction of prediction which should
            intersect with ``intersect_with``. Defaults to 0.1.
        crop_from (str | sitk.Image | None, optional): whether the
            input should be cropped centered on a given mask object. Defaults to None.
        crop_padding (tuple[int, int, int] | None, optional): padding to be
            added to the cropped region. Defaults to None.
        cascade_mode (str, optional): whether to crop inputs to consecutive bounding boxes
            or to intersect consecutive outputs. Defaults to "intersect".
    """

    def coherce_to_list(obj: Any, n: int) -> list[Any] | tuple[Any]:
        if isinstance(obj, (list, tuple)):
            if len(obj) != n:
                raise ValueError(f"{obj} should have length {n}")
        else:
            obj = [obj for _ in range(n)]
        return obj

    output_dir = output_dir.strip().rstrip("/")

    if isinstance(series_paths, (tuple, list)) is False:
        raise ValueError(
            f"series_paths should be list of strings or list of list of strings (is {series_paths})"
        )
    if isinstance(nnunet_path, (list, tuple)):
        # minimal input parsing
        series_paths_list = None
        class_idx_list = None
        if isinstance(series_paths, (tuple, list)):
            if isinstance(series_paths[0], (list, tuple)):
                series_paths_list = series_paths
            elif isinstance(series_paths[0], str):
                series_paths_list = [series_paths for _ in nnunet_path]
        if isinstance(class_idx, int) or class_idx is None:
            class_idx_list = [class_idx for _ in nnunet_path]
        elif isinstance(class_idx, (tuple, list)):
            class_idx_list = class_idx
        proba_threshold = coherce_to_list(proba_threshold, len(nnunet_path))
        min_confidence = coherce_to_list(min_confidence, len(nnunet_path))

        if series_paths_list is None:
            raise ValueError(
                f"series_paths should be list of strings or list of list of strings (is {series_paths})"
            )

        logger.info("Using nnunet_path %s for inference", nnunet_path)
        logger.info("Using series_paths %s for inference", series_paths)
        logger.info("Using class_idx_list %s for inference", class_idx_list)
        logger.info("Using proba_threshold %s for inference", proba_threshold)
        logger.info("Using min_confidence %s for inference", min_confidence)
        logger.info("Using checkpoint_name %s for inference", checkpoint_name)

        all_volumes, all_good_file_paths = load_series(series_paths, is_dicom)
        all_predictions = []
        all_proba_maps = []
        for i in range(len(nnunet_path)):
            if i == (len(nnunet_path) - 1):
                out = output_dir
            else:
                out = os.path.join(tmp_dir, f"stage_{i}")
            mask, proba_map = single_model_inference(
                predictor=predictor,
                nnunet_path=nnunet_path[i].strip(),
                volumes=all_volumes[i],
                class_idx=class_idx_list[i],
                checkpoint_name=checkpoint_name.strip(),
                output_dir=out,
                tmp_dir=os.path.join(tmp_dir, f"stage_{i}"),
                use_folds=use_folds,
                proba_threshold=proba_threshold[i],
                min_confidence=min_confidence[i],
                intersect_with=intersect_with,
                min_overlap=min_overlap,
                crop_from=crop_from,
                crop_padding=crop_padding,
            )
            all_predictions.append(mask)
            all_proba_maps.append(proba_map)
            if i < (len(nnunet_path) - 1):
                if cascade_mode == "intersect":
                    logger.info("Using mask for intersection")
                    intersect_with = mask
                elif cascade_mode == "crop":
                    logger.info("Using mask for cropping")
                    crop_from = mask
        # keep first from last predicted series to replicate previous behaviour
        if is_dicom:
            good_file_paths = [all_good_file_paths[series_paths[-1][0]]]
        else:
            good_file_paths = None
    else:
        mask, proba_map = single_model_inference(
            predictor=predictor,
            nnunet_path=nnunet_path.strip(),
            series_paths=series_paths,
            checkpoint_name=checkpoint_name.strip(),
            output_dir=output_dir,
            tmp_dir=tmp_dir,
            use_folds=use_folds,
            proba_threshold=proba_threshold,
            min_confidence=min_confidence,
            intersect_with=intersect_with,
            min_overlap=min_overlap,
            crop_from=crop_from,
            crop_padding=crop_padding,
        )
        all_predictions = [mask]
        all_proba_maps = [proba_map]

    logger.info("Finished inference")

    return all_predictions, all_proba_maps, good_file_paths, all_volumes


def export_predictions(
    masks: list[sitk.Image],
    output_dir: str,
    volumes: list[list[sitk.Image]] | None = None,
    proba_maps: list[list[sitk.Image]] | None = None,
    good_file_paths: list[str] | None = None,
    suffix: str | None = None,
    is_dicom: bool = False,
    metadata: dict[str, str] | None = None,
    save_proba_map: bool = False,
    save_nifti_inputs: bool = False,
    save_rt_struct_output: bool = False,
):
    output_names = {
        "prediction": (
            "prediction" if suffix is None else f"prediction_{suffix}"
        ),
        "probabilities": (
            "probabilities" if suffix is None else f"proba_{suffix}"
        ),
        "struct": "struct" if suffix is None else f"struct_{suffix}",
    }

    output_paths = {}
    stage_dirs = [
        os.path.join(output_dir, f"stage_{i}") for i in range(len(masks))
    ]
    for stage_dir in stage_dirs:
        os.makedirs(stage_dir, exist_ok=True)

    if is_dicom is True:
        if metadata is None:
            raise ValueError("metadata must be defined when is_dicom is True")
        seg_writer = SegWriter.init_from_metadata_dict(metadata)

    mask_paths = []
    for i, mask in enumerate(masks):
        output_nifti_path = (
            f"{stage_dirs[i]}/{output_names['prediction']}.nii.gz"
        )
        sitk.WriteImage(mask, output_nifti_path)
        mask_paths.append(output_nifti_path)
        logger.info("Exported prediction mask %d to %s", i, output_nifti_path)
    output_paths["nifti_prediction"] = mask_paths

    if save_proba_map is True:
        proba_map_paths = []
        for i, proba_map in enumerate(proba_maps):
            output_nifti_path = (
                f"{stage_dirs[i]}/{output_names['probabilities']}.nii.gz"
            )
            sitk.WriteImage(proba_map, output_nifti_path)
            proba_map_paths.append(output_nifti_path)
            logger.info(
                "Exported probability map %d to %s", i, output_nifti_path
            )
        output_paths["nifti_proba"] = proba_map_paths

    if save_nifti_inputs is True:
        niftis = []
        for i, volume_set in enumerate(volumes):
            for volume in volume_set:
                output_nifti_path = os.path.join(
                    stage_dirs[i], "input_volume.nii.gz"
                )
                sitk.WriteImage(volume, output_nifti_path)
                niftis.append(output_nifti_path)
                logger.info(
                    "Exported input volume %d to %s", i, output_nifti_path
                )
        output_paths["nifti_inputs"] = niftis

    if is_dicom is True:
        if metadata is None:
            raise ValueError("if is_dicom is True metadata must be specified")
        dicom_seg_paths = []
        dicom_struct_paths = []
        for i, mask in enumerate(masks):
            dcm_seg_output_path = (
                f"{stage_dirs[i]}/{output_names['prediction']}.dcm"
            )
            status = seg_writer.write_dicom_seg(
                mask,
                source_files=good_file_paths[0],
                output_path=dcm_seg_output_path,
            )
            if "empty" in status:
                logger.info("Mask %d is empty, skipping DICOMseg/RTstruct", i)
            elif save_rt_struct_output:
                dcm_rts_output_path = (
                    f"{stage_dirs[i]}/{output_names['struct']}.dcm"
                )
                status = seg_writer.write_dicom_rtstruct(
                    mask,
                    source_files=good_file_paths[0],
                    output_path=dcm_rts_output_path,
                )
                dicom_seg_paths.append(dcm_seg_output_path)
                dicom_struct_paths.append(dcm_rts_output_path)
                logger.info(
                    "Exported DICOM struct %d to %s", i, dicom_struct_paths[-1]
                )
            else:
                dicom_seg_paths.append(dcm_seg_output_path)
                logger.info(
                    "Exported DICOM segmentation %d to %s",
                    i,
                    dicom_seg_paths[-1],
                )
        output_paths["dicom_segmentation"] = dicom_seg_paths
        if save_rt_struct_output:
            output_paths["dicom_struct"] = dicom_struct_paths

        if save_proba_map is True:
            dicom_proba_paths = []
            for i, proba_map in enumerate(proba_maps):
                output_path = (
                    f"{stage_dirs[i]}/{output_names['probabilities']}.dcm"
                )
                seg_writer.write_dicom_seg(
                    proba_map,
                    source_files=good_file_paths[0],
                    output_path=output_path,
                    is_fractional=True,
                )
                dicom_proba_paths.append(output_path)
                logger.info(
                    "Exported DICOM fractional segmentation %d to %s",
                    i,
                    dicom_proba_paths[-1],
                )
            output_paths["dicom_fractional_segmentation"] = dicom_proba_paths

    return output_paths
