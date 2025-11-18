import requests
import os
from requests.auth import HTTPBasicAuth
from zipfile import ZipFile

ORTHANC_URL = os.environ.get("ORTHANC_URL", "http://localhost:8042")
ORTHANC_USER = os.environ.get("ORTHANC_USER", None)
ORTHANC_PASSWORD = os.environ.get("ORTHANC_PASSWORD", None)
TMP_STUDY_DIR = os.environ.get("TMP_STUDY_DIR", "/tmp/nnunet_serve/orthanc")

if ORTHANC_USER and ORTHANC_PASSWORD:
    AUTH = HTTPBasicAuth(ORTHANC_USER, ORTHANC_PASSWORD)
else:
    AUTH = None


def get_all_patients():
    """
    Get all patients from Orthanc.

    Returns:
        A list of patients.
    """
    response = requests.get(f"{ORTHANC_URL}/patients", auth=AUTH)
    response.raise_for_status()
    return response.json()


def get_all_studies():
    """
    Get all studies from Orthanc.

    Returns:
        A list of studies.
    """
    response = requests.get(f"{ORTHANC_URL}/studies", auth=AUTH)
    response.raise_for_status()
    return response.json()


def get_all_series():
    """
    Get all series from Orthanc.

    Returns:
        A list of series.
    """
    response = requests.get(f"{ORTHANC_URL}/series", auth=AUTH)
    response.raise_for_status()
    return response.json()


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


def download_series(series_id: str | list[str], output_dir: str | None = None):
    """
    Download a series from Orthanc.

    Args:
        series_id (str | list[str]): The ID of the series or a list of series IDs.
        output_dir (str | None): The directory to save the series. If None, the
            series will be saved in a temporary directory (TMP_STUDY_DIR).
    """

    if isinstance(series_id, list):
        series_id = [series_id]

    response = requests.get(
        f"{ORTHANC_URL}/series/{series_id}/archive", auth=AUTH
    )
    response.raise_for_status()

    if output_dir is None:
        output_dir = TMP_STUDY_DIR
    study_id = get_series(series_id)["ParentStudy"]
    patient_id = get_study(study_id)["ParentPatient"]
    real_out_dir = os.path.join(output_dir, patient_id, study_id)
    os.makedirs(real_out_dir, exist_ok=True)

    all_paths = {}
    for s_id in series_id:
        with open(os.path.join(real_out_dir, f"{s_id}.zip"), "wb") as f:
            f.write(response.content)

        with ZipFile(os.path.join(real_out_dir, f"{s_id}.zip"), "r") as zip_ref:
            zip_ref.extractall(real_out_dir)

        os.remove(os.path.join(real_out_dir, f"{s_id}.zip"))

        all_paths[s_id] = os.path.join(real_out_dir, s_id)

    return all_paths


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
    return response.json()
