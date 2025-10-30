import os
import uuid
import shutil
import zipfile
import tarfile
from pathlib import Path
from fastapi import HTTPException
from fastapi import UploadFile

ALLOWED_EXTENSIONS = {".zip", ".tar", ".gz", ".tgz"}

NNUNET_OUTPUT_DIR = Path(os.environ.get("NNUNET_OUTPUT_DIR", "/tmp/nnunet"))


def get_study_path(job_id: str) -> Path:
    """
    Returns the path to the study directory.

    Args:
        job_id (str): The job ID.

    Returns:
        Path: The path to the study directory.
    """
    return NNUNET_OUTPUT_DIR / job_id


def store_uploaded_file(upload: UploadFile, job_id: str | None = None) -> Path:
    """
    Saves the uploaded file to a unique temporary directory and returns the
    absolute path to that directory.  The caller can then pass this path to
    the existing inference code that expects a filesystem location.

    Args:
        upload (UploadFile): The uploaded file.

    Returns:
        Path: The absolute path to the temporary directory.
    """
    if job_id is None:
        job_id = uuid.uuid4().hex
    work_dir = get_study_path(job_id) / "inputs"
    work_dir.mkdir(parents=True, exist_ok=True)

    dest_path = work_dir / upload.filename

    if dest_path.suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {dest_path.suffix}",
        )

    with dest_path.open("wb") as out_file:
        shutil.copyfileobj(upload.file, out_file)

    try:
        if dest_path.suffix == ".zip":
            with zipfile.ZipFile(dest_path) as zf:
                zf.extractall(work_dir)
        else:
            with tarfile.open(dest_path) as tf:
                tf.extractall(work_dir)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to unpack archive: {exc}",
        )
    dest_path.unlink()
    return work_dir


def zip_directory(src_dir: Path) -> Path:
    """
    Compress *src_dir* (which contains the inference outputs) into a temporary
    zip file and return the path to that zip.
    The caller is responsible for deleting the zip after it has been sent.
    """
    zip_path = src_dir.with_suffix(".zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in src_dir.rglob("*"):
            if file_path.is_file():
                # store relative path inside the zip
                zf.write(file_path, file_path.relative_to(src_dir.parent))
    return zip_path
