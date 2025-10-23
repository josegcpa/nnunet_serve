from totalsegmentator.dicom_utils import load_snomed_mapping

TASK_CONVERSION = {
    "total_fast": 297,
    "total_fastest": 298,
    "total": [291, 292, 293, 294, 295],
    "total_mr_fast": 852,
    "total_mr_fastest": 853,
    "total_mr": [850, 851],
    "lung_vessels": 258,
    "cerebral_bleed": 150,
    "hip_implant": 260,
    "body_fast": 300,
    "body": 299,
    "body_mr_fast": 598,  # todo: train
    "body_mr": 597,
    "vertebrae_mr": 756,
    "pleural_pericard_effusion": 315,
    "liver_vessels": 8,
    "head_glands_cavities": 775,
    "headneck_bones_vessels": 776,
    "head_muscles": 777,
    "headneck_muscles": [778, 779],
    "oculomotor_muscles": 351,
    "lung_nodules": 913,
    "kidney_cysts": 789,
    "breasts": 527,
    "ventricle_parts": 552,
    "liver_segments": 570,
    "liver_segments_mr": 576,
    "craniofacial_structures": 115,
    "abdominal_muscles": 952,
    "teeth": 113,
    # Commercial models
    "vertebrae_body": 305,
    "heartchambers_highres": 301,
    "appendicular_bones": 304,
    "appendicular_bones_mr": 855,
    "tissue_types": 481,
    "tissue_types_mr": 925,
    "tissue_4_types": 485,
    "face": 303,
    "face_mr": 856,
    "brain_structures": 409,
    "thigh_shoulder_muscles": 857,
    "thigh_shoulder_muscles_mr": 857,
    "coronary_arteries": 507,
    "aortic_sinuses": 920,
}

ADDITIONAL_SNOMED_TYPES = {
    "liver_segment_1": ("CouinaudHepaticSegmentI", "71133005"),
    "liver_segment_2": ("CouinaudHepaticSegmentII", "277956007"),
    "liver_segment_3": ("CouinaudHepaticSegmentIII", "277957003"),
    "liver_segment_4": ("CouinaudHepaticSegmentIV", "277958008"),
    "liver_segment_5": ("CouinaudHepaticSegmentV", "277959000"),
    "liver_segment_6": ("CouinaudHepaticSegmentVI", "277960005"),
    "liver_segment_7": ("CouinaudHepaticSegmentVII", "277961009"),
    "liver_segment_8": ("CouinaudHepaticSegmentVIII", "277962002"),
}


def load_snomed_mapping_expanded():
    anatomical_structure = {
        "scheme": "SCT",
        "value": "123037004",
        "meaning": "Anatomical Structure",
    }
    snomed_mapping = load_snomed_mapping()
    for k in ADDITIONAL_SNOMED_TYPES:
        snomed_mapping[k] = {
            "property_category": anatomical_structure,
            "property_type": {
                "scheme": "SCT",
                "meaning": ADDITIONAL_SNOMED_TYPES[k][0],
                "code": ADDITIONAL_SNOMED_TYPES[k][1],
            },
        }

    for k in snomed_mapping:
        if "property_type" in snomed_mapping[k]:
            if "value" in snomed_mapping[k]["property_type"]:
                snomed_mapping[k]["property_type"]["code"] = snomed_mapping[k][
                    "property_type"
                ]["value"]

    return snomed_mapping
