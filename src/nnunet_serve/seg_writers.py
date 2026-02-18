"""
Utilities for writing segmentations.
"""

import logging
import os
import random
import re
from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import highdicom as hd
import numpy as np
import pydicom
import SimpleITK as sitk
from matplotlib import colormaps
from pydicom import DataElement
from pydicom.sr.codedict import Code, codes
from nnunet_serve.pydicom_seg_template import rgb_to_cielab, from_dcmqi_metainfo
from tqdm import tqdm

from nnunet_serve.coding import (
    CATEGORY_MAPPING,
    CATEGORY_CONCEPTS,
    CODING_SCHEME_INFORMATION,
    CODING_SCHEME_INFORMATION_VR,
    LATERALITY_CODING,
    NATURAL_LANGUAGE_TO_CODE,
)
from nnunet_serve.logging_utils import get_logger
from nnunet_serve.utils import sort_dicom_slices
from nnunet_serve.str_processing import get_laterality, to_camel_case

logger = get_logger(__name__)

DEFAULT_SEGMENT_SCHEME = os.environ.get("DEFAULT_SEGMENT_SCHEME", "SCT")
N_FRACTIONAL = 10


def strip_laterality(name: str) -> str:
    name = re.sub("[ _]*[lL]eft[ _]*", "", name)
    name = re.sub("[ _]*[rR]ight[ _]*", "", name)
    name = name.strip()
    return name


def process_name(name: str) -> str:
    name = strip_laterality(name)
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
                "Only SCT and EUCAIM schemes are supported for automatic retrieval"
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
        meaning=strip_laterality(to_camel_case(segment_dict["name"], " ")),
        scheme_designator=segment_dict["scheme"],
    )

    if segment_dict["category_number"] is None:
        segment_dict["category_number"] = CATEGORY_MAPPING[
            segment_dict["scheme"]
        ]["type"][str(segment_code.value)]

    laterality = get_laterality(segment_dict["label"])
    if not laterality:
        laterality = get_laterality(segment_dict["name"])
    segment_dict["laterality"] = laterality
    if laterality:
        if str(segment_dict["category_number"]) == "49755003":
            pass
        elif laterality.lower() not in segment_dict["label"].lower():
            segment_dict["label"] = f'{laterality} {segment_dict["label"]}'
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
        category_concepts = CATEGORY_CONCEPTS[DEFAULT_SEGMENT_SCHEME]
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
            category_code = [
                category_concepts[k]
                for k in category_concepts
                if category_concepts[k].value == segment_dict["category_number"]
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
            if segment_dict["laterality"] is not None:
                lat_code = LATERALITY_CODING[segment_dict["scheme"]][
                    segment_dict["laterality"]
                ]
                segment_description.SegmentedPropertyTypeModifierCodeSequence = [
                    lat_code
                ]
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

    def to_array_if_necessary(
        self, mask: np.ndarray | sitk.Image
    ) -> np.ndarray:
        if isinstance(mask, sitk.Image):
            mask = sitk.GetArrayFromImage(mask)
        return mask

    def make_compliant(self, f):
        if hasattr(f, "PatientSex") is False:
            f.PatientSex = "O"
        if hasattr(f, "AccessionNumber") is False:
            f.AccessionNumber = ""
        if hasattr(f, "StudyID") is False:
            f.StudyID = ""
        return f

    def write_dicom_seg(
        self,
        mask_array: np.ndarray | sitk.Image,
        source_files: list[str],
        output_path: str,
        is_fractional: bool = False,
        is_fractional_compliant: bool = False,
    ):
        """
        Writes a DICOM segmentation file.

        Args:
            mask_array (np.ndarray | sitk.Image): the mask array or sitk image.
            source_files (list[str]): the list of DICOM source files (as returned
                by ``nnunet_serve.utils.read_dicom_as_sitk``).
            output_path (str): the output path.
            is_fractional (bool, optional): whether the mask is fractional.
                Defaults to False.
            is_fractional_compliant (bool, optional): whether the probability mask
                should be converted to a map with ``N_FRACTIONAL`` labels,
                each corresponding to a percentage.
        """
        mask_array = self.to_array_if_necessary(mask_array)
        sorted_source_files = sort_dicom_slices(list(source_files))
        if sorted_source_files != list(source_files):
            idx_map = {p: i for i, p in enumerate(source_files)}
            try:
                order = [idx_map[p] for p in sorted_source_files]
            except KeyError:
                order = None
            if order is not None:
                mask_array = mask_array[order, ...]
            source_files = sorted_source_files
        # adjust array size and segment descriptions to the strictly necessary
        labels = np.unique(mask_array)
        labels = labels[labels > 0]
        if len(labels) == 0:
            logger.warning("Mask is empty")
            return "empty"
        segment_descriptions = []
        if is_fractional is False and is_fractional_compliant is False:
            label_dict = {label: i + 1 for i, label in enumerate(labels)}
            label_dict[0] = 0
            mask_array = np.vectorize(label_dict.get)(mask_array)
            for i, label in enumerate(labels.astype(int)):
                seg_d = deepcopy(self.segment_descriptions[label - 1])
                seg_d.SegmentNumber = i + 1
                segment_descriptions.append(seg_d)
            if len(mask_array.shape) != 4:
                mask_array = one_hot_encode(
                    mask_array, len(segment_descriptions)
                )
        else:
            segment_descriptions = self.segment_descriptions
            if mask_array.ndim == 3:
                mask_array = mask_array[..., None]
        image_datasets = [
            self.make_compliant(hd.imread(str(f))) for f in source_files
        ]

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
        label_meanings = []
        for s in segment_descriptions:
            ss = s.SegmentedPropertyTypeCodeSequence[0]
            if isinstance(ss, pydicom.Dataset):
                meaning = ss.CodeMeaning
            else:
                meaning = ss.meaning
            label_meanings.append(meaning)
        seg_series_description = "Seg of " + ", ".join(label_meanings)
        if len(seg_series_description) > 64:
            seg_series_description = seg_series_description[:61] + "..."
        if is_fractional_compliant:
            if len(segment_descriptions) > 1:
                raise ValueError(
                    "is_fractional_compliant==True requires single label"
                )
            seg_type = hd.seg.SegmentationTypeValues.BINARY
            logger.info("Converting mask to pseudo-fractional DICOM seg")
            sd = segment_descriptions[0]
            jet_cmap = colormaps.get("jet")
            percents = np.linspace(0, 1, N_FRACTIONAL + 1, endpoint=True)
            colours = jet_cmap(percents)
            label = sd.SegmentLabel
            if hasattr(sd, "SegmentDescription"):
                desc = sd.SegmentDescription
            else:
                desc = None
            new_segment_descriptions = []
            new_mask_array = []
            for i in range(N_FRACTIONAL):
                new_sd = deepcopy(sd)
                p1, p2 = percents[i], percents[i + 1]
                new_sd.SegmentNumber = i + 1
                new_label = f"{label} ({int(p1*100)}-{int(p2*100)}%)"
                new_sd.SegmentLabel = new_label
                if desc is not None:
                    new_desc = f"{desc} ({int(p1*100)}-{int(p2*100)}%)"
                    new_sd.SegmentDescription = new_desc
                curr_mask = (mask_array > p1) & (mask_array <= p2)
                new_sd.RecommendedDisplayCIELabValue = rgb_to_cielab(
                    colours[i, :3] * 255
                )
                new_segment_descriptions.append(new_sd)
                new_mask_array.append(curr_mask)
            segment_descriptions = new_segment_descriptions
            mask_array = np.concatenate(new_mask_array, axis=-1).astype(bool)
            logger.info("Converted mask to pseudo-fractional DICOM seg")
        elif is_fractional:
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
            series_description=seg_series_description,
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
        metadata = from_dcmqi_metainfo(metadata_file)
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


def export_predictions(
    masks: list[sitk.Image],
    output_dir: str,
    volumes: list[list[sitk.Image]] | None = None,
    proba_maps: list[list[sitk.Image]] | None = None,
    good_file_paths: list[str] | None = None,
    suffix: str | None = None,
    is_dicom: bool = False,
    seg_writers: SegWriter | list[SegWriter] | None = None,
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
            if proba_map is None:
                logger.warning(f"proba_map for stage {i} is None, skipping")
                proba_map_paths.append(None)
                continue
            sitk.WriteImage(proba_map, output_nifti_path)
            proba_map_paths.append(output_nifti_path)
            logger.info(
                "Exported probability map %d to %s", i, output_nifti_path
            )
        output_paths["nifti_proba"] = proba_map_paths

    if save_nifti_inputs is True:
        niftis = []
        for i, volume_set in enumerate(volumes):
            for j, volume in enumerate(volume_set):
                output_nifti_path = os.path.join(
                    stage_dirs[i], f"input_volume_{j}.nii.gz"
                )
                sitk.WriteImage(volume, output_nifti_path)
                niftis.append(output_nifti_path)
                logger.info(
                    "Exported input volume %d to %s for stage %d",
                    j,
                    output_nifti_path,
                    i,
                )
        output_paths["nifti_inputs"] = niftis

    if is_dicom is True:
        dicom_seg_paths = []
        dicom_struct_paths = []
        for i, mask in enumerate(masks):
            dcm_seg_output_path = (
                f"{stage_dirs[i]}/{output_names['prediction']}.dcm"
            )
            status = seg_writers[i].write_dicom_seg(
                mask,
                source_files=good_file_paths[0],
                output_path=dcm_seg_output_path,
            )
            if "empty" in status:
                logger.info("Mask %d is empty, skipping DICOMseg/RTstruct", i)
                dicom_seg_paths.append(None)
                dicom_struct_paths.append(None)
            elif save_rt_struct_output:
                dcm_rts_output_path = (
                    f"{stage_dirs[i]}/{output_names['struct']}.dcm"
                )
                status = seg_writers[i].write_dicom_rtstruct(
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
            logger.info("Exporting probabilities")
            for i, proba_map in enumerate(proba_maps):
                if proba_map is None:
                    dicom_proba_paths.append(None)
                    continue
                output_path = (
                    f"{stage_dirs[i]}/{output_names['probabilities']}.dcm"
                )
                status = seg_writers[i].write_dicom_seg(
                    proba_map,
                    source_files=good_file_paths[0],
                    output_path=output_path,
                    is_fractional_compliant=True,
                )
                if status == "empty":
                    dicom_proba_paths.append(None)
                    logger.info(
                        f"Mask {i} is empty, skipping DICOM probabilities."
                    )
                    continue
                dicom_proba_paths.append(output_path)
                logger.info(
                    "Exported DICOM fractional segmentation %d to %s",
                    i,
                    dicom_proba_paths[-1],
                )
            output_paths["dicom_fractional_segmentation"] = dicom_proba_paths

    logger.info("Finished exporting predictions")
    return output_paths


if __name__ == "__main__":
    seg_writer = SegWriter(
        segment_names=[{"name": "Liver", "number": 1, "label": "Liver"}],
        algorithm_name="nnUNet",
        algorithm_version="v1.0",
    )

    print(seg_writer.segment_descriptions)
