# Configuring models

Model configuration makes use of `model-serve-spec.yaml`. This is a relatively simple YAML file where each model is defined, together with potential aliases and the relevant paths.

- **`model_folder`**: absolute path where models exist or will be downloaded (for TotalSegmentator tasks).
- **`models[]`**: list of model entries. Each entry can define:
  - **`id`**: identifier used in API requests (see `nnunet_id`).
  - **`rel_path`**: substring pattern to locate the model directory under `model_folder` (folder containing `fold_0`, etc.).
  - **`name`** and optional **`aliases`**: user-friendly names/aliases; all map to `id`.
  - **`metadata`**: DICOM metadata for DICOM-SEG/RTStruct export. Either:
    - `{ path: <path/to/metadata.json> }` to a DCMQI template file (please see: [this web app](https://qiicr.org/dcmqi/#/seg)), or
    - an inline object with keys such as `algorithm_name`, `segment_names`, etc. (see examples in `model-serve-spec.yaml`). When both are provided, `metadata.path` takes precedence.
  - **`min_mem`**: minimum free GPU memory in MiB to start (`wait_for_gpu`).
  - **`is_totalseg`**: boolean flag to indicate if the model is a TotalSegmentator model. This is important as there are some peculariaties to TotalSegmentator models that are handled differently (e.g., weights are auto-downloaded and `metadata` is auto-derived).
  - **`default_args`**: defaults for request parameters (e.g., `series_folders`, `use_folds`, `proba_threshold`, `min_confidence`, `tta`, `save_proba_map`, `checkpoint_name`, etc.). When multiple models are requested, list-valued defaults are merged per model (see `get_default_params()` in `nnunet_serve_utils.py`).
  - For TotalSegmentator tasks, you can specify `totalseg_task` (e.g., `total_fastest`); weights are auto-downloaded and `metadata` is auto-derived.

## Environment variables

* `NNUNET_OUTPUT_DIR`: path used to store temporary files. Defaults to "/tmp/nnunet".
* `LOGS_DIR`: path used to store logs. Defaults to "./logs".
* `PORT`: port used by the API. Defaults to "12345".
* `MAX_REQUESTS_PER_MINUTE`: maximum number of requests per minute. Defaults to "10".
* `ORTHANC_URL`: URL of the Orthanc server. Defaults to `http://localhost:8042`.
* `ORTHANC_USER`: username used to authenticate with Orthanc. Defaults to `None`.
* `ORTHANC_PASSWORD`: password used to authenticate with Orthanc. Defaults to `None`.
* `TMP_STUDY_DIR`: path used to store temporary study files (if downloads or similar are necessary). Defaults to `/tmp/nnunet_serve/orthanc`.
* `DEFAULT_SEGMENT_SCHEME`: default segment scheme used for DICOM-SEG/RTStruct export. Defaults to `SCT` (SNOMED-CT).
* `NNUNET_SERVE_LOGGING_LEVEL`: logging level used by the API. Defaults to `INFO`.
* `TOTALSEG_WEIGHTS_PATH`: path to the TotalSegmentator weights directory. Defaults to `<model-serve-spec.yaml["model_folder"]>/totalseg`.
* `MODEL_SERVE_SPEC`: path to the model serve specification file. Defaults to `model-serve-spec.yaml`.
* `DEBUG`: whether to run the API in debug mode (avoids using try/except blocks and produces errors which are easier to trace). Defaults to `False`.
 