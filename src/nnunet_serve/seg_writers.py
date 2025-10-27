"""
Utilities for writing segmentations.
"""

import logging
import random
import os
import re
from dataclasses import dataclass
from pathlib import Path
from difflib import SequenceMatcher


import highdicom as hd
import numpy as np
import pydicom
import pydicom_seg
import SimpleITK as sitk
from copy import deepcopy
from pydicom import DataElement
from pydicom.sr.codedict import Code, codes
from pydicom_seg.template import rgb_to_cielab
from tqdm import tqdm

from nnunet_serve.coding import (
    CATEGORY_MAPPING,
    CODING_SCHEME_INFORMATION,
    CODING_SCHEME_INFORMATION_VR,
    NATURAL_LANGUAGE_TO_CODE,
)
from nnunet_serve.str_processing import to_camel_case
from nnunet_serve.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_SEGMENT_SCHEME = os.environ.get("DEFAULT_SEGMENT_SCHEME", "SCT")


def process_name(name: str) -> str:
    name = re.sub("[ _]*[lL]eft[ _]*", "", name)
    name = re.sub("[ _]*[rR]ight[ _]*", "", name)
    name = re.sub("[ _]+", "", name)
    name = name.strip()
    return name


def get_segment_type_code(segment: dict | str, i: int) -> Code:
    """
    Resolve and build the DICOM coded concept for a segment.

    Args:
        segment (dict | str): Segment specification. If a dict, the following
            keys are supported:

            - name (str): Human-readable name of the structure (e.g., "Liver").
            - number (int, optional): 1-based segment number. Defaults to
              ``i + 1`` at the call site.
            - label (str, optional): Display label for the segment. Defaults to
              the value of ``name``.
            - tracking_id (str, optional): Stable identifier used for DICOM
              Tracking ID. Defaults to ``"Segment{number}_{label}"``.
            - code (str | int | None, optional): Coded value (Code Value) for
              the segmented property type. If ``None``, it will be looked up
              automatically from the concept dictionary based on ``name``.
            - scheme (str, optional): Coding Scheme Designator. Only ``"SCT"``
              is supported for automatic lookup. Defaults to ``"SCT"``.
            - category_number (str | int | None, optional): Code Value for the
              segmented property category. If ``None``, it will be derived
              downstream using ``CATEGORY_MAPPING`` using the output ``code.value``
              property as key. Even when this is provided, it should represent
              a SNOMED CT category number.

            If a string is provided, it is interpreted as ``name``; the
            remaining fields are inferred using the defaults above. This
            segment string will be used to index
            ``pydicom.sr._concepts_dict.CONCEPTS``, after being preprocessed
            to remove laterality-based indicators (e.g., "Right", "Left"; these
            are nonetheless included in the ``segment_dict["label"]``).
        - i (int): 1-based segment number.

    Returns:
        - Code: A ``pydicom.sr.codedict.Code`` representing the segment's
            type (value, meaning, scheme_designator).
        - dict: The normalized segment dictionary with all defaults and the
            resolved code populated.

    Raises:
        ValueError: If ``segment`` is neither ``dict`` nor ``str``, if
            automatic lookup is requested with a non-"SCT" scheme, or if the
            concept cannot be found in the dictionary (closest matches are
            reported).
    """
    segment_dict = {}
    if isinstance(segment, dict):
        segment_dict["name"] = segment["name"]
        segment_dict["number"] = segment.get("number", i + 1)
        segment_dict["label"] = segment.get("label", segment["name"])
        segment_dict["tracking_id"] = segment.get(
            "tracking_id",
            f"Segment{segment_dict['number']}_{segment_dict['label']}",
        )
        segment_dict["code"] = segment.get("code", None)
        segment_dict["scheme"] = segment.get("scheme", "SCT")
        segment_dict["category_number"] = segment.get("category_number", None)
    elif isinstance(segment, str):
        segment_dict["name"] = segment
        segment_dict["number"] = i + 1
        segment_dict["label"] = segment
        segment_dict[
            "tracking_id"
        ] = f"Segment{segment_dict['number']}_{segment_dict['label']}"
        segment_dict["code"] = None
        segment_dict["scheme"] = DEFAULT_SEGMENT_SCHEME
        segment_dict["category_number"] = None
    else:
        raise ValueError(f"Invalid segment: {segment}")
    if segment_dict["code"] is None:
        name = to_camel_case(process_name(segment_dict["name"]))
        if segment_dict["scheme"] not in ["SCT", "EUCAIM"]:
            raise ValueError(
                f"Only SCT and EUCAIM schemes are supported for automatic retrieval"
            )
        if name not in NATURAL_LANGUAGE_TO_CODE[segment_dict["scheme"]]:
            closest_matches = []
            for k in NATURAL_LANGUAGE_TO_CODE[segment_dict["scheme"]]:
                if close_match(k, name, 0.8):
                    closest_matches.append(k)
            raise ValueError(
                f"Segment {name} not found in {segment_dict['scheme']}. Closest matches: {closest_matches}"
            )
        code_info = NATURAL_LANGUAGE_TO_CODE[segment_dict["scheme"]][name]
        segment_dict["name"] = code_info[0]
        segment_dict["code"] = code_info[1]
    segment_code = Code(
        value=segment_dict["code"],
        meaning=segment_dict["name"],
        scheme_designator=segment_dict["scheme"],
    )
    return segment_code, segment_dict


def get_empty_segment_description(
    algorithm_type, algorithm_identification, tracking_id: str
):
    """
    Returns an empty segment description.

    Args:
        algorithm_type (str): algorithm type.
        algorithm_identification (str): algorithm identification.
        tracking_id (str): tracking ID.

    Returns:
        hd.seg.SegmentDescription: empty segment description.
    """
    return hd.seg.SegmentDescription(
        segment_number=99,
        segment_label="Empty segment",
        segmented_property_category=codes.CID7150.PhysicalObject,
        segmented_property_type=codes.CID7150.PhysicalObject,
        algorithm_type=algorithm_type,
        algorithm_identification=algorithm_identification,
        tracking_uid=hd.UID(),
        tracking_id=tracking_id,
    )


def random_color_generator():
    """
    Returns a random color as a tuple of RGB values.

    Returns:
        tuple: tuple of RGB values.
    """
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)


def close_match(a, b, ratio: float = 0.8) -> bool:
    """
    Returns True if the ratio of matching characters between strings a and b
    is greater than the specified ratio.

    Args:
        a (str): first string.
        b (str): second string.
        ratio (float): ratio of matching characters.

    Returns:
        bool: True if the ratio of matching characters between strings a and b
            is greater than the specified ratio.
    """
    return SequenceMatcher(None, a, b).ratio() > ratio


def one_hot_encode(arr: np.ndarray, n_labels: int) -> np.ndarray:
    """
    Converts a numpy array to a one-hot encoded numpy array.

    Args:
        arr (np.ndarray): numpy array to be converted.

    Returns:
        np.ndarray: one-hot encoded numpy array.
    """
    output_arr = np.zeros([*arr.shape, n_labels])
    for i in range(1, n_labels + 1):
        output_arr[..., i - 1] = arr == i
    output_arr = output_arr
    return output_arr


def save_mask_as_rtstruct(
    img_data: np.ndarray,
    dcm_reference_file: str,
    output_path: str,
    segment_info: list[tuple[str, list[int]]],
) -> None:
    """
    Converts a numpy array to an RT (radiotherapy) struct object. Could be a
        multi-class object (each n > 0 corresponds to a class). The number of
        classes corresponds to ``np.unique(img_data).shape[0] - 1``.

    Args:
        img_data (np.ndarray): numpy array with n non-zero unique values, each
            of which corresponds to a class.
        dcm_reference_file (str): reference DICOM files.
        output_path (str): output file for RT struct file.
        segment_info (tuple[str, list[int]]): segment information. Should be a
            list with size equal to the number of classes, and each element
            should be a tuple whose first element is the segment description
            and the second element a list of RGB values.
    """
    try:
        from rt_utils import RTStructBuilder
    except ImportError:
        raise ImportError(
            "rt_utils is required to save masks in RT struct format"
        )
    # based on the TotalSegmentator implementation

    logging.basicConfig(level=logging.WARNING)  # avoid messages from rt_utils

    # create new RT Struct - requires original DICOM
    rtstruct = RTStructBuilder.create_new(dicom_series_path=dcm_reference_file)

    # retrieve selected classes
    img_data = img_data.swapaxes(0, 2)
    selected_classes = np.unique(img_data)
    selected_classes = selected_classes[selected_classes > 0].tolist()
    if len(selected_classes) == 0:
        return None

    # add mask to RT Struct
    for class_idx in tqdm(selected_classes):
        class_name, class_colour = segment_info[class_idx - 1]
        binary_img = img_data == class_idx
        if binary_img.sum() > 0:  # only save none-empty images
            # add segmentation to RT Struct
            rtstruct.add_roi(
                mask=binary_img,  # has to be a binary numpy array
                name=class_name,
                color=class_colour,
            )

    rtstruct.save(str(output_path))


@dataclass
class SegWriter:
    algorithm_name: str
    segment_names: list[str | dict[str, str]] | None = None
    segment_descriptions: list[hd.seg.SegmentDescription] | None = None
    algorithm_version: str = "v1.0"
    algorithm_family: pydicom.sr.coding.Code = (
        codes.CID7162.ArtificialIntelligence
    )
    algorithm_type: hd.seg.SegmentAlgorithmTypeValues = (
        hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC
    )
    instance_number: int = 1
    series_number: int = 999
    manufacturer: str = "Algorithm"
    manufacturer_model_name: str = "AlgorithmModel"
    series_description: str = "Segmentation"
    clinical_trial_series_id: str = "1"
    clinical_trial_time_point_id: str = "1"
    body_part_examined: str = "BODY"
    validate: bool = False

    def __post_init__(self):
        if self.segment_descriptions is None and self.segment_names is None:
            raise ValueError(
                "Either segment_descriptions or segment_names must be provided"
            )
        category_concepts = codes.CID7150.concepts
        self.algorithm_identification = hd.AlgorithmIdentificationSequence(
            name=self.algorithm_name,
            version=self.algorithm_version,
            family=self.algorithm_family,
        )
        if self.segment_names is None:
            return
        if self.segment_descriptions is None:
            self.segment_descriptions = []
        for i, segment in enumerate(self.segment_names):
            type_code, segment_dict = get_segment_type_code(segment, i)

            if segment_dict["category_number"] is None:
                category_number = CATEGORY_MAPPING[segment_dict["scheme"]][
                    "type"
                ][str(type_code.value)]
            category_code = [
                category_concepts[k]
                for k in category_concepts
                if category_concepts[k].value == category_number
            ][0]
            random_rgb_colour = random_color_generator()
            segment_description = hd.seg.SegmentDescription(
                segment_number=segment_dict["number"],
                segment_label=segment_dict["label"],
                segmented_property_category=category_code,
                segmented_property_type=type_code,
                algorithm_type=self.algorithm_type,
                algorithm_identification=self.algorithm_identification,
                tracking_uid=hd.UID(),
                tracking_id=segment_dict["tracking_id"],
            )
            segment_description.RecommendedDisplayCIELabValue = rgb_to_cielab(
                random_rgb_colour
            )
            csi = CODING_SCHEME_INFORMATION[segment_dict["scheme"]]
            for (
                prop_type
            ) in segment_description.SegmentedPropertyTypeCodeSequence:
                for k in csi:
                    prop_type[k] = DataElement(
                        tag=k,
                        VR=CODING_SCHEME_INFORMATION_VR[k],
                        value=csi[k],
                    )
            for (
                prop_cat
            ) in segment_description.SegmentedPropertyCategoryCodeSequence:
                for k in csi:
                    prop_cat[k] = DataElement(
                        tag=k,
                        VR=CODING_SCHEME_INFORMATION_VR[k],
                        value=csi[k],
                    )
            self.segment_descriptions.append(segment_description)

        if self.validate is True:
            for i, segment_description in enumerate(self.segment_descriptions):
                logger.info(
                    "Segment %s: %s, %s, class_idx=%i",
                    segment_description.segment_number,
                    segment_description.segment_label,
                    segment_description.segmented_property_type.meaning,
                    i,
                )
                logger.info(
                    segment_description.SegmentedPropertyTypeCodeSequence
                )

    def to_array_if_necessary(
        self, mask: np.ndarray | sitk.Image
    ) -> np.ndarray:
        if isinstance(mask, sitk.Image):
            mask = sitk.GetArrayFromImage(mask)
        return mask

    def write_dicom_seg(
        self,
        mask_array: np.ndarray | sitk.Image,
        source_files: list[str],
        output_path: str,
        is_fractional: bool = False,
    ):
        mask_array = self.to_array_if_necessary(mask_array)[::-1, :, :]
        # adjust array size and segment descriptions to the scritly necessary
        labels = np.unique(mask_array)
        labels = labels[labels > 0]
        if len(labels) == 0:
            logging.warning("Mask is empty")
            return "empty"
        label_dict = {label: i + 1 for i, label in enumerate(labels)}
        label_dict[0] = 0
        mask_array = np.vectorize(label_dict.get)(mask_array)
        segment_descriptions = []
        for i, label in enumerate(labels):
            seg_d = deepcopy(self.segment_descriptions[label - 1])
            seg_d.SegmentNumber = i + 1
            segment_descriptions.append(seg_d)

        if len(mask_array.shape) != 4:
            mask_array = one_hot_encode(mask_array, len(segment_descriptions))
        image_datasets = [hd.imread(str(f)) for f in source_files]

        if hasattr(image_datasets[0], "InstanceNumber"):
            image_datasets = sorted(
                image_datasets, key=lambda x: x.InstanceNumber
            )
        elif hasattr(image_datasets[0], "ImagePositionPatient"):
            image_datasets = sorted(
                image_datasets, key=lambda x: float(x.ImagePositionPatient[2])
            )

        first = image_datasets[0]
        rows, cols = int(getattr(first, "Rows")), int(getattr(first, "Columns"))
        frames = len(image_datasets)
        if (frames, rows, cols) != mask_array.shape[:-1]:
            raise Exception(
                f"Mask shape {mask_array.shape} does not match image shape {frames}x{rows}x{cols}"
            )
        if mask_array.shape[-1] != len(segment_descriptions):
            raise Exception(
                f"Mask shape {mask_array.shape} does not match number of segments {len(segment_descriptions)}"
            )

        # Create the Segmentation instance
        if is_fractional:
            seg_type = hd.seg.SegmentationTypeValues.FRACTIONAL
        else:
            seg_type = hd.seg.SegmentationTypeValues.BINARY
        seg_dataset = hd.seg.Segmentation(
            source_images=image_datasets,
            pixel_array=mask_array,
            segmentation_type=seg_type,
            segment_descriptions=segment_descriptions,
            series_instance_uid=hd.UID(),
            series_number=999,
            sop_instance_uid=hd.UID(),
            instance_number=1,
            manufacturer=self.manufacturer,
            manufacturer_model_name=self.manufacturer_model_name,
            software_versions=self.algorithm_version,
            device_serial_number="42",
            series_description="Segmentation",
        )
        seg_dataset.ClinicalTrialSeriesID = self.clinical_trial_series_id
        seg_dataset.ClinicalTrialTimePointID = self.clinical_trial_time_point_id
        seg_dataset.BodyPartExamined = self.body_part_examined

        seg_dataset.save_as(output_path)

        return "success"

    def write_dicom_rtstruct(
        self,
        mask_array: np.ndarray | sitk.Image,
        source_files: list[str],
        output_path: str,
    ):
        mask_array = self.to_array_if_necessary(mask_array)
        segment_info = [
            [s.SegmentLabel, list(random_color_generator())]
            for s in self.segment_descriptions
        ]
        save_mask_as_rtstruct(
            mask_array,
            Path(source_files[0]).parent,
            output_path,
            segment_info,
        )

        return "success"

    @staticmethod
    def init_from_dcmqi_metadata_file(
        metadata_file: str,
        algorithm_version: str = "v1.0",
        manufacturer: str = "Algorithm",
        manufacturer_model_name: str = "AlgorithmModel",
    ):
        metadata = pydicom_seg.template.from_dcmqi_metainfo(metadata_file)
        segments = list(metadata.SegmentSequence)
        algorithm_name = segments[0].SegmentAlgorithmName
        algorithm_type = hd.seg.SegmentAlgorithmTypeValues[
            segments[0].SegmentAlgorithmType
        ]
        series_description = metadata.SeriesDescription
        clinical_trial_series_id = metadata.ClinicalTrialSeriesID
        clinical_trial_time_point_id = metadata.ClinicalTrialTimePointID
        body_part_examined = metadata.BodyPartExamined
        return SegWriter(
            segment_descriptions=segments,
            algorithm_name=algorithm_name,
            algorithm_version=algorithm_version,
            algorithm_family=codes.CID7162.ArtificialIntelligence,
            algorithm_type=algorithm_type,
            instance_number=1,
            series_number=999,
            series_description=series_description,
            manufacturer=manufacturer,
            manufacturer_model_name=manufacturer_model_name,
            clinical_trial_series_id=clinical_trial_series_id,
            clinical_trial_time_point_id=clinical_trial_time_point_id,
            body_part_examined=body_part_examined,
        )

    @staticmethod
    def init_from_metadata_dict(
        metadata: dict[str, str], validate: bool = False
    ):
        if "path" in metadata:
            return SegWriter.init_from_dcmqi_metadata_file(metadata["path"])
        else:
            return SegWriter(**metadata, validate=validate)


if __name__ == "__main__":
    seg_writer = SegWriter(
        segment_names=[{"name": "Liver", "number": 1, "label": "Liver"}],
        algorithm_name="nnUNet",
        algorithm_version="v1.0",
    )

    print(seg_writer.segment_descriptions)
