import os
import shutil
import tarfile
import uuid
import re
import zipfile
from pathlib import Path

from fastapi import HTTPException, UploadFile

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


def _print_tree(file_dict: dict, indent: int = 0, is_last: bool = False) -> str:
    def get_indent_character(indent):
        out = ""
        if indent > 1:
            out += "  " * (indent - 1)
        if indent > 0:
            if is_last:
                out += "└─"
            else:
                out += "├─"
        return out

    output = get_indent_character(indent) + file_dict["parent"] + "\n"
    children = file_dict["children"]
    indent_ex = indent + 1
    n = len(children)
    for i, c in enumerate(children):
        if isinstance(c, str):
            output += get_indent_character(indent_ex) + c + "\n"
        elif isinstance(c, dict):
            output += _print_tree(c, indent_ex, i == n - 1)

    return output


def _list_files_recursive(
    directory: str,
    regex_pattern: str | None = None,
    parent_dir: str | None = None,
) -> dict[str, list[str | dict]]:
    """
    Recursively lists the files in a directory and outputs nested dictionaries
    with {"parent": str, "children": list[dict | str]}.

    Args:
        directory (str): directory which will be recursively listed.
        regex_pattern (str): regex_pattern which can be used for filtering files.

    Returns:
    """

    def files_str(files: list[str]) -> str:
        n = len(files)
        if n > 5:
            file_str = ", ".join(files[:5]) + "..."
        else:
            file_str = ", ".join(files)
        return f"{n} files ({file_str})"

    output_dict = {"parent": directory, "children": []}
    if parent_dir:
        dir_to_list = os.path.join(parent_dir, directory)
    else:
        dir_to_list = directory
    files = os.listdir(dir_to_list)
    if regex_pattern:
        regex = re.compile(regex_pattern)
        files = [f for f in files if regex.match(f)]
    for i in range(len(files)):
        full_path = os.path.join(dir_to_list, files[i])
        if os.path.isdir(full_path):
            files[i] = f"{files[i]}/"
            output_dict["children"].append(
                _list_files_recursive(files[i], regex_pattern, dir_to_list)
            )
        else:
            output_dict["children"].append(files[i])

    files = []
    dirs = []
    for c in output_dict["children"]:
        if isinstance(c, str):
            files.append(c)
        else:
            dirs.append(c)
    output_dict["children"] = []
    if len(files) > 0:
        output_dict["children"].append(files_str(files))
    output_dict["children"].extend(dirs)
    return output_dict
