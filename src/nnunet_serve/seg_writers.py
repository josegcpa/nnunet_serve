"""
Utilities for writing segmentations.
"""

import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path

import highdicom as hd
import numpy as np
import pydicom
import pydicom_seg
import SimpleITK as sitk
from copy import deepcopy
from pydicom.sr._concepts_dict import concepts as CONCEPTS
from pydicom.sr.codedict import Code, codes
from tqdm import tqdm

from nnunet_serve.category_mapping import CATEGORY_MAPPING
from nnunet_serve.logging_utils import get_logger

logger = get_logger(__name__)


def get_empty_segment(
    algorithm_type, algorithm_identification, tracking_id: str
):
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
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)


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
    segment_names: list[str | dict[str, str]]
    algorithm_name: str
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

    def __post_init__(self):
        self.segment_descriptions = []
        category_concepts = codes.CID7150.concepts
        self.algorithm_identification = hd.AlgorithmIdentificationSequence(
            name=self.algorithm_name,
            version=self.algorithm_version,
            family=self.algorithm_family,
        )
        for i, segment in enumerate(self.segment_names):
            segment_dict = {}
            if isinstance(segment, dict):
                segment_dict["name"] = self.process_name(segment["name"])
                segment_dict["number"] = segment.get("number", i + 1)
                segment_dict["label"] = segment.get("label", segment["name"])
                segment_dict["tracking_id"] = segment.get(
                    "tracking_id",
                    f"Segment{segment_dict['number']}_{segment_dict['label']}",
                )
                segment_dict["code"] = segment.get("code", None)
            elif isinstance(segment, str):
                segment_dict["name"] = self.process_name(segment)
                segment_dict["number"] = i + 1
                segment_dict["label"] = segment
                segment_dict[
                    "tracking_id"
                ] = f"Segment{segment_dict['number']}_{segment_dict['label']}"
                segment_dict["code"] = None
            else:
                raise ValueError(f"Invalid segment: {segment}")
            if segment_dict["code"] is None:
                type_code_dict = CONCEPTS["SCT"][segment_dict["name"]]
                segment_dict["code"] = list(type_code_dict.keys())[0]
                segment_dict["name"], _ = type_code_dict[segment_dict["code"]]
            type_code = Code(
                value=segment_dict["code"],
                meaning=segment_dict["name"],
                scheme_designator="SCT",
            )

            category_number = CATEGORY_MAPPING["type"][str(type_code.value)]
            category_code = [
                category_concepts[k]
                for k in category_concepts
                if category_concepts[k].value == category_number
            ][0]

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
            self.segment_descriptions.append(segment_description)

    def process_name(self, name: str) -> str:
        name = re.sub("[ ]*[lL]eft[ ]*", "", name)
        name = re.sub("[ ]*[rR]ight[ ]*", "", name)
        name = name.strip()
        return name

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
        mask_array = self.to_array_if_necessary(mask_array)
        # adjust array size and segment descriptions to the scritly necessary
        labels = np.unique(mask_array)
        labels = labels[labels > 0]
        if len(labels) == 0:
            logging.warning("Mask is empty")
            return "empty"
        label_dict = {label: i + 1 for i, label in enumerate(labels)}
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
            segment_names=segments,
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
    def init_from_metadata_dict(metadata: dict[str, str]):
        if "path" in metadata:
            return SegWriter.init_from_dcmqi_metadata_file(metadata["path"])
        else:
            return SegWriter(**metadata)


if __name__ == "__main__":
    seg_writer = SegWriter(
        segment_names=[{"name": "Liver", "number": 1, "label": "Liver"}],
        algorithm_name="nnUNet",
        algorithm_version="v1.0",
    )

    print(seg_writer.segment_descriptions)
