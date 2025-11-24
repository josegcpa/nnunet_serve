import requests
import os
from typing import Any
from io import BytesIO
from requests.auth import HTTPBasicAuth
from zipfile import ZipFile
from fastmcp.exceptions import ToolError

ORTHANC_URL = os.environ.get("ORTHANC_URL", "http://localhost:8042")
ORTHANC_USER = os.environ.get("ORTHANC_USER", None)
ORTHANC_PASSWORD = os.environ.get("ORTHANC_PASSWORD", None)
TMP_STUDY_DIR = os.environ.get("TMP_STUDY_DIR", "/tmp/nnunet_serve/orthanc")

if ORTHANC_USER and ORTHANC_PASSWORD:
    AUTH = HTTPBasicAuth(ORTHANC_USER, ORTHANC_PASSWORD)
else:
    AUTH = None

try:
    requests.get(ORTHANC_URL, auth=AUTH)
    ORTHANC_AVAILABLE = True
except requests.exceptions.RequestException as e:
    ORTHANC_AVAILABLE = False


def fail_if_orthanc_not_available(func):
    def decorator(*args, **kwargs):
        if not ORTHANC_AVAILABLE:
            raise ToolError("Orthanc is not available")
        return func(*args, **kwargs)

    return decorator


@fail_if_orthanc_not_available
def get_all_patients():
    """
    Get all patients from Orthanc.

    Returns:
        A list of patients.
    """
    response = requests.get(f"{ORTHANC_URL}/patients", auth=AUTH)
    response.raise_for_status()
    return response.json()


@fail_if_orthanc_not_available
def get_all_studies():
    """
    Get all studies from Orthanc.

    Returns:
        A list of studies.
    """
    response = requests.get(f"{ORTHANC_URL}/studies", auth=AUTH)
    response.raise_for_status()
    return response.json()


@fail_if_orthanc_not_available
def get_all_series():
    """
    Get all series from Orthanc.

    Returns:
        A list of series.
    """
    response = requests.get(f"{ORTHANC_URL}/series", auth=AUTH)
    response.raise_for_status()
    return response.json()


@fail_if_orthanc_not_available
def get_series_in_study(study_id: str):
    """
    Get all series in a study from Orthanc.

    Args:
        study_id: The ID of the study.

    Returns:
        A list of series.
    """
    response = requests.get(
        f"{ORTHANC_URL}/studies/{study_id}/series", auth=AUTH
    )
    response.raise_for_status()
    return response.json()


@fail_if_orthanc_not_available
def get_patient(patient_id: str):
    """
    Get a patient from Orthanc.

    Args:
        patient_id: The ID of the patient.

    Returns:
        A patient.
    """
    response = requests.get(f"{ORTHANC_URL}/patients/{patient_id}", auth=AUTH)
    response.raise_for_status()
    return response.json()


@fail_if_orthanc_not_available
def get_study(study_id: str):
    """
    Get a study from Orthanc.

    Args:
        study_id: The ID of the study.

    Returns:
        A study.
    """
    response = requests.get(f"{ORTHANC_URL}/studies/{study_id}", auth=AUTH)
    response.raise_for_status()
    return response.json()


@fail_if_orthanc_not_available
def get_series(series_id: str):
    """
    Get a series from Orthanc.

    Args:
        series_id: The ID of the series.

    Returns:
        A series.
    """
    response = requests.get(f"{ORTHANC_URL}/series/{series_id}", auth=AUTH)
    response.raise_for_status()
    return response.json()


@fail_if_orthanc_not_available
def download_series(series_id: str | list[str], output_dir: str | None = None):
    """
    Download a series from Orthanc.

    Args:
        series_id (str | list[str]): The ID of the series or a list of series IDs.
        output_dir (str | None): The directory to save the series. If None, the
            series will be saved in a temporary directory (TMP_STUDY_DIR).
    """

    if isinstance(series_id, str):
        series_id = [series_id]

    if output_dir is None:
        output_dir = TMP_STUDY_DIR
    os.makedirs(output_dir, exist_ok=True)

    all_paths = {}
    for s_id in series_id:
        response = requests.get(
            f"{ORTHANC_URL}/series/{s_id}/archive", auth=AUTH
        )
        response.raise_for_status()

        with ZipFile(BytesIO(response.content)) as zip_ref:
            members = zip_ref.namelist()
            zip_ref.extractall(output_dir)

        all_paths[s_id] = os.path.join(output_dir, os.path.dirname(members[0]))
    return all_paths


@fail_if_orthanc_not_available
def upload_instance(instance_path: str):
    """
    Upload an instance to Orthanc.

    Args:
        instance_path (str): The path to the instance.
    """
    response = requests.post(
        f"{ORTHANC_URL}/instances",
        auth=AUTH,
        files={"file": open(instance_path, "rb")},
    )
    response.raise_for_status()
    return response.text


@fail_if_orthanc_not_available
def get_series_for_series_uid(
    series_uid: str,
) -> dict[str, Any] | None:
    """
    Returns a dictionary with the series for a given series UID.

    Args:
        series_uid (str): The series UID.

    Returns:
        A dictionary with the series for the given series UID.
    """
    response = requests.post(
        f"{ORTHANC_URL}/tools/find",
        json={
            "Level": "Series",
            "Query": {"SeriesInstanceUID": series_uid},
            "Expand": True,
        },
        auth=AUTH,
    )
    response.raise_for_status()
    response_json = response.json()
    if len(response_json) == 0:
        return None
    series = {}
    for series in response_json:
        sid = series["ID"]
        series[sid] = {
            "series_dicom_tags": series.get("MainDicomTags", {}),
        }
    return series


@fail_if_orthanc_not_available
def get_all_series_for_study_uid(
    study_uid: str,
) -> dict[str, Any] | None:
    """
    Returns a dictionary with the series for a given study UID.

    Args:
        study_uid (str): The study UID.

    Returns:
        A dictionary with the series for the given study UID.
    """
    response = requests.post(
        f"{ORTHANC_URL}/tools/find",
        json={
            "Level": "Studies",
            "Query": {"StudyInstanceUID": study_uid},
            "Expand": True,
        },
        auth=AUTH,
    )
    response.raise_for_status()
    response_json = response.json()
    if len(response_json) == 0:
        return None
    studies = {}
    for study in response_json:
        cid = study["ID"]
        studies[cid] = {
            "patient_dicom_tags": study.get("PatientMainDicomTags", {}),
            "study_dicom_tags": study.get("MainDicomTags", {}),
            "series": {},
        }
        all_series = study.get("Series", [])
        for sid in all_series:
            series = get_series(sid)
            dicom_tags = series.get("MainDicomTags", {})
            if "ImageOrientationPatient" in dicom_tags:
                ori = dicom_tags["ImageOrientationPatient"]
                dicom_tags["ImageOrientationPatient"] = [
                    float(x) for x in ori.split("\\")
                ]
            studies[cid]["series"][sid] = {"series_dicom_tags": dicom_tags}
    return studies


@fail_if_orthanc_not_available
def get_all_studies_for_patient_id(
    patient_id: str,
) -> dict[str, Any] | None:
    response = requests.post(
        f"{ORTHANC_URL}/tools/find",
        json={
            "Level": "Studies",
            "Query": {"PatientID": patient_id},
            "Expand": True,
        },
        auth=AUTH,
    )
    response.raise_for_status()
    response_json = response.json()
    if len(response_json) == 0:
        return None
    studies = {}
    for study in response_json:
        cid = study["ID"]
        studies[cid] = {
            "patient_dicom_tags": study.get("PatientMainDicomTags", {}),
            "study_dicom_tags": study.get("MainDicomTags", {}),
            "series": {},
        }
        all_series = study.get("Series", [])
        for sid in all_series:
            series = get_series(sid)
            dicom_tags = series.get("MainDicomTags", {})
            if "ImageOrientationPatient" in dicom_tags:
                ori = dicom_tags["ImageOrientationPatient"]
                dicom_tags["ImageOrientationPatient"] = [
                    float(x) for x in ori.split("\\")
                ]
            studies[cid]["series"][sid] = {"series_dicom_tags": dicom_tags}
    return studies


def get_series_tags(series_id: str) -> dict[str, Any]:
    """
    Returns the tags for the first instance of a given series ID.

    Args:
        series_id (str): The ID of the series.

    Returns:
        dict[str, Any]: The tags for the series.
    """
    exclude_names = [
        "AccessionNumber",
        "InstanceNumber",
        "DeidentificationMethodCodeSequence",
        "ClinicalTrialTimePointID",
        "DeidentificationMethod",
        "InstanceNumber",
        "AcquisitionNumber",
    ]
    first_instance = get_series(series_id)["Instances"][0]
    response = requests.get(
        f"{ORTHANC_URL}/instances/{first_instance}/tags", auth=AUTH
    )
    response.raise_for_status()
    response_json = response.json()
    output = {}
    for k in response_json:
        name = response_json[k]["Name"]
        if "UID" in name or name in exclude_names:
            continue
        output[name] = response_json[k]["Value"]
    return output
