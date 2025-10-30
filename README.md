# Docker container-ready nnUNet wrapper for SITK-readable and DICOM files

## Context

Given that [nnUNet](https://github.com/MIC-DKFZ/nnUNet) is a relatively flexible framework, we have developed a container that allows users to run nnUNet in a container while varying the necessary models. The main features are inferring all necessary parameters from the nnUNet files (spacing, extensions) and working for both DICOM folder and SITK-readable files. If the input is a DICOM, the segmentation is converted into a DICOM-seg file, compatible with PACS systems.

## Usage

### Configure models: `model-serve-spec.yaml`

Model configuration makes use of `model-serve-spec.yaml`. This is a relatively simple YAML file where each model is defined, together with potential aliases and the relevant paths.

- **`model_folder`**: absolute path where models exist or will be downloaded (for TotalSegmentator tasks).
- **`models[]`**: list of model entries. Each entry can define:
  - **`id`**: identifier used in API requests (see `nnunet_id`).
  - **`rel_path`**: substring pattern to locate the model directory under `model_folder` (folder containing `fold_0`, etc.).
  - **`name`** and optional **`aliases`**: user-friendly names/aliases; all map to `id`.
  - **`metadata`**: DICOM metadata for DICOM-SEG/RTStruct export. Either:
    - `{ path: <path/to/metadata.json> }` to a DCMQI template file, or
    - an inline object with keys such as `algorithm_name`, `segment_names`, etc. (see examples in `model-serve-spec.yaml`). When both are provided, `metadata.path` takes precedence.
  - **`min_mem`**: minimum free GPU memory in MiB to start (`wait_for_gpu`).
  - **`default_args`**: defaults for request parameters (e.g., `series_folders`, `use_folds`, `proba_threshold`, `min_confidence`, `tta`, `save_proba_map`, `checkpoint_name`, etc.). When multiple models are requested, list-valued defaults are merged per model (see `get_default_params()` in `nnunet_serve_utils.py`).
  - For TotalSegmentator tasks, you can specify `totalseg_task` (e.g., `total_fastest`); weights are auto-downloaded and `metadata` is auto-derived.


### Standalone script

A considerable objective of this framework was its deployment as a standalone tool (for `bash`). To use it:

1. Install the necessary packages using an appropriate Python environment (i.e. `pip install -r requirements.txt`). We have tested this using Python `v3.11`
2. Run `uv run nnunet_serve.entrypoints.entrypoint.py --help` to see the available options
3. Segment away!

```bash
uv run nnunet_serve.entrypoints.entrypoint.py --help
```

```
options:
  -h, --help            show this help message and exit
  --study_path, -i STUDY_PATH
                        Path to input series
  --series_folders, -s SERIES_FOLDERS [SERIES_FOLDERS ...]
                        Path to input series folders
  --nnunet_id NNUNET_ID [NNUNET_ID ...]
                        nnUNet ID
  --checkpoint_name CHECKPOINT_NAME
                        Checkpoint name for nnUNet
  --output_dir, -o OUTPUT_DIR
                        Path to output directory
  --folds, -f FOLDS [FOLDS ...]
                        Sets which folds should be used with nnUNet
  --tta, -t             Uses test-time augmentation during prediction
  --tmp_dir TMP_DIR     Temporary directory
  --is_dicom, -D        Assumes input is DICOM (and also converts to DICOM seg; prediction.dcm in output_dir)
  --proba_map, -p       Produces a Nifti format probability map (probabilities.nii.gz in output_dir)
  --proba_threshold PROBA_THRESHOLD [PROBA_THRESHOLD ...]
                        Sets probabilities in proba_map lower than proba_threhosld to 0
  --min_confidence MIN_CONFIDENCE [MIN_CONFIDENCE ...]
                        Removes objects whose max prob is smaller than min_confidence
  --rt_struct_output    Produces a DICOM RT Struct file (struct.dcm in output_dir; requires DICOM input)
  --save_nifti_inputs, -S
                        Moves Nifti inputs to output folder (volume_XXXX.nii.gz in output_dir)
  --cascade_mode {intersect,crop} [{intersect,crop} ...]
                        Defines the cascade mode. Must be either intersect or crop.
  --intersect_with INTERSECT_WITH
                        Calculates the IoU with the SITK mask image in this path and uses this value to filter images such that IoU <
                        --min_intersection are ruled out.
  --min_intersection MIN_INTERSECTION [MIN_INTERSECTION ...]
                        Minimum intersection over the union to keep a candidate.
  --crop_from CROP_FROM
                        Crops the input to the bounding box of the SITK mask image in this path.
  --crop_padding CROP_PADDING [CROP_PADDING ...]
                        Padding to be added to the cropped region.
  --class_idx CLASS_IDX [CLASS_IDX ...]
                        Class index.
  --suffix SUFFIX       Adds a suffix (_suffix) to the outputs if specified.
```

Example:

The example below outlines the path to a given study (`--study_path`) and to a given series folder (`--series_folders`). The `--nnunet_id` flag outlines the models to be used, in this case, `prostate` and `prostate_zones` (the two models are applied sequentially, and the output from the first model is used to crop the input to the second model as noted in `--cascade_mode`). The `--output_dir` flag outlines the path to the output directory. The `--is_dicom` flag outlines that the input is a DICOM file. The `--proba_threshold` flag outlines the probability threshold for the probability map. The `--cascade_mode` flag outlines the cascade mode (`crop` or `intersect`). The `--save_nifti_inputs` flag outlines that the Nifti inputs should be saved to the output directory. The `--crop_padding` flag outlines the padding to be added to the cropped region.

```bash
uv run nnunet_serve.entrypoints.entrypoint \
  --study_path path/to/study \
  --series_folders relative/path/to/series \
  --nnunet_id prostate prostate_zones \
  --output_dir path/to/output \
  --is_dicom \
  --proba_threshold None \
  --cascade_mode crop \
  --save_nifti_inputs \
  --crop_padding 20 20 20
```

#### Extended argument description

A core concept underlies this framework - that of cascading predictions. The output of a prediction is used as an input for the next prediction by either cropping the input image, filtering objects in the output image based on a minimum intersection or by appending it to the input image. Fields flagged with 💧 support multiple values in compliance with the cascade. For these fields, multiple space-separated values can be specified as long as the number of values matches the number of models (`nnunet_id`) in the cascade. In some instances, multiple values at each stage might require specification (`series_folders` or `folds`). For these, at each stage, multiple values can be specified using commas (`,`).

- **`--study_path` / `-i`**: Path to the input study directory containing the imaging data. Required.
- **`--series_folders` / `-s`**: One or more relative paths to series folders within the study. Required. Multiple **space separated** values refer to multiple stages of the cascade. At each stage, different series can be specified using commas (`,`) 💧
- **`--nnunet_id`**: Identifier(s) of the nnU‑Net model(s) to run. Provide one or more model names; they will be applied sequentially 💧
- **`--checkpoint_name`**: Name(s) of the checkpoint file(s) to load (default: `checkpoint_final.pth`) 💧
- **`--output_dir` / `-o`**: Directory where all output files (segmentations, maps, logs) will be written. Required.
- **`--folds` / `-f`**: Which cross‑validation folds to use. Accepts a list of integers (default: `0`). Multiple **space separated** values refer to multiple stages of the cascade; multiple values at each stage can be specified using commas (`,`) 💧
- **`--tta` / `-t`**: Enable test‑time augmentation (mirroring) during inference.
- **`--tmp_dir`**: Temporary directory for intermediate files (default: `.tmp`).
- **`--is_dicom` / `-D`**: Indicate that the input series are DICOM. The tool will also generate a DICOM segmentation (`prediction.dcm`).
- **`--proba_map` / `-p`**: Output a probability map in NIfTI format (`probabilities.nii.gz`).
- **`--proba_threshold`**: Threshold applied to the probability map; values below this are set to zero (default: `0.5`). Can be a list to match multiple models. 💧
- **`--min_confidence`**: Minimum confidence required for a predicted object; objects below this are discarded (default: none). Can be a list. 💧
- **`--rt_struct_output`**: Produce a DICOM RT Struct file (`struct.dcm`) in the output directory (requires DICOM input).
- **`--save_nifti_inputs` / `-S`**: Save the NIfTI versions of the input volumes in the output folder.
- **`--cascade_mode`**: Define how multiple models are combined: `intersect` (default), `crop` or `concatenate`.
- **`--intersect_with`**: Path to a mask image used to compute IoU; predictions with IoU below `--min_intersection` are removed.
- **`--min_intersection`**: Minimum IoU required to keep a candidate when using `--intersect_with` (default: `0.1`).  When using `--cascade_mode intersect`, this flag is used for intersection filtering.
- **`--crop_from`**: Path to a mask image whose bounding box will be used to crop the input before the next model. 
- **`--crop_padding`**: Padding (in voxels) added around the cropped region (default: `10 10 10`).  When using `--cascade_mode crop` this flag is used for cropping.
- **`--class_idx`**: Index or list of class indices to retain in the final output (default: `all`). 💧
- **`--suffix`**: Optional suffix appended to output filenames (e.g., `_v1`).

### Logging and status updates for CLI (for `uv run nnunet_serve.entrypoints.entrypoint_prod`)

To facilitate integration into production environments, we have added a logging function to `entrypoint_prod.py`. This works by specifying the following CLI arguments:

- `--update_url` - this is the URL to be used to post job status. Will post `--job_id` (under `job_id`), `--success_message` or `--failure_message` depending on the outcome of the job (under `status`). Errors are logged using `output_error` and any additional information is logged under `output_log`. In other words, the following JSON is posted to `--update_url`:

```json
{
    "job_id": <job_id>,
    "status": <"success_message" or "failure_message">,
    "output_error": <error message>,
    "output_log": <log message>
}
```

- `--success_message` - specifies the success message
- `--failure_message` - specifies the failure message
- `--job_id` - specifies the job ID to be used to post job status
- `--log_file` - specifies the path to a log file to be created. This file will contain the job ID, the success/failure message, and the output log. If `log_file` already exists, only `status`, `output_error` and `output_log` are updated, while `job_id` is only added to the log if it has not already been specified in the pre-existing `log_file`.

### Notes on using DICOM

It is necessary to generate metadata templates for the conversion between the segmentation prediction volume and DICOM volumes. To generate these, the `pydicom_seg` developers recommend [this web app](https://qiicr.org/dcmqi/#/seg). It is easy to use and generates reliable metadata templates. Metadata templates should be generated for all segmentation targets to ensure that everything is correctly formatted.

## API

This repository includes a FastAPI server that exposes nnU-Net inference as an HTTP API. The server is implemented in `src/nnunet_serve/nnunet_serve.py` and configured by `model-serve-spec.yaml`.

### Run the server

- **Local (development):**

```bash
# optionally set the port via env var (defaults to 12345)
export NNUNET_SERVE_PORT=12345

python -m uvicorn nnunet_serve.nnunet_serve:create_app \
  --host 0.0.0.0 \
  --port ${NNUNET_SERVE_PORT} \
  --reload
```

- **Environment variables:**
  - `MODEL_SERVE_SPEC`: path to a model serve spec file. Defaults to `model-serve-spec.yaml` in the working directory.
  - `TOTALSEG_WEIGHTS_PATH`: optional override for where TotalSegmentator weights are downloaded/cached. Defaults to `<model_folder>/totalseg` based on `model-serve-spec.yaml`.
  - `NNUNET_SERVE_PORT`: the port the server listens on (default: `12345`).

Ensure your `model-serve-spec.yaml` is present and correctly references your models. GPU and `nvidia-smi` must be available; the server waits for a GPU with enough free memory before running a job.

### Endpoints

- **`GET /model_info`**
  - Returns the server’s model registry resolved from `model-serve-spec.yaml` and the filesystem.
  - Each entry is keyed by `id` and includes, when found, `path`, `rel_path`, `name`, `metadata`, `min_mem`, `default_args`, `n_classes`, `model_information` (parsed from `dataset.json`), and flags like `is_totalseg`.

- **`GET /request-params`**
  - Returns the JSON schema of the request body for `/infer` (Pydantic model `InferenceRequest`). Use this to dynamically build clients and as documentation.

- **`POST /infer`**
  - Runs inference for one or multiple models.
  - Response includes execution metadata and exported file paths (see below). On errors, a JSON with `status="failed"` and `error` is returned with HTTP 400/500.

- **`POST /infer_file`**
  - Accepts a file (or archive) upload, stores it, builds an `InferenceRequest`, and delegates to `/infer`. Returns a job ID and inference result. The fields that should be sent are identical to those of the `/infer` method with the exception of the `study_path` field, which is not required.

- **`GET /download/{job_id}`**
  - Serves the zip file containing the inference outputs for the given job ID. Returns 404 if the job ID is unknown or the file has been cleaned up.

- **`GET /healthz`**
  - Simple health check endpoint returning `{"status": "ok"}`.

- **`GET /readyz`**
  - Readiness probe indicating whether models are loaded and a GPU is available (if required). Returns `{ "status": "ok" }` when ready, otherwise `{ "status": "starting" }` with additional fields.

### Request body schema (InferenceRequest)

Required fields:
- **`nnunet_id`**: string or list of strings. Must match a model `id`, `name`, or any alias from `model-serve-spec.yaml`.
- **`study_path`**: string path to the study root directory.
- **`series_folders`**:
  - Single model: list of relative series folder names under `study_path`.
  - Multiple models: list of lists, one per model, each a list of relative series folder names under `study_path`.
  - DICOM inputs (`is_dicom=true`): each entry must point to a directory containing a single DICOM series (not a study root). For multi-series inputs per model (e.g., T2/DWI/ADC), additional series are rigidly resampled to the first series’ geometry for inference.
- **`output_dir`**: directory where outputs will be written.

Common optional fields (with server defaults or per-model `default_args`):
- **`class_idx`**: integer or list of integers per model. Keeps only selected classes in outputs and probability maps.
- **`checkpoint_name`**: checkpoint filename in each model folder. Default `checkpoint_final.pth` (or from `default_args`).
- **`tmp_dir`**: temp directory. Default `.tmp`.
- **`is_dicom`**: boolean. If true, reads DICOM series and exports DICOM-SEG/RTStruct using model `metadata`. Default `false`.
- **`tta`**: boolean. If true, enables mirroring. Default `true`.
- **`use_folds`**: list of ints. Default `[0]` unless overridden.
- **`proba_threshold`**: float or list of floats per model; required if `save_proba_map=true`.
- **`min_confidence`**: float or list of floats per model; filters candidate components.
- **`intersect_with`**: path to a mask image to intersect candidates; see `min_intersection`.
- **`min_intersection`**: float IoU threshold for candidate filtering. Default `0.1`.
- **`crop_from`**: path to a mask used to crop inputs by bounding box. See `crop_padding` and `cascade_mode`.
- **`crop_padding`**: tuple of three ints. Default `(10, 10, 10)`.
- **`cascade_mode`**: string, one of `intersect` or `crop`. Default `intersect`.
- Export controls:
  - **`save_proba_map`**: boolean. If true, exports probability maps. Requires `proba_threshold` not null.
  - **`save_nifti_inputs`**: boolean. If true and `is_dicom=true`, exports input volumes as NIfTI.
  - **`save_rt_struct_output`**: boolean. If true and `is_dicom=true`, also exports RT Struct.
  - **`suffix`**: string appended to output filenames (e.g., `prediction_<suffix>.nii.gz`).

Notes:
- For multi-model requests (`nnunet_id` is a list), `series_folders` must be a list of lists of the same length, and list-valued parameters (`class_idx`, `proba_threshold`, etc.) can be supplied per model. Defaults are merged accordingly.
- When `is_dicom=true`, each model must have `metadata` defined in `model-serve-spec.yaml` (either `path` to a DCMQI JSON template or an inline metadata object). Otherwise inference will fail.

### Response schema (POST /infer)

On success (HTTP 200):
- **`time_elapsed`**: seconds to complete the request.
- **`gpu`**: CUDA device index used.
- **`nnunet_path`**: string or list of model paths used.
- **`metadata`**: metadata object(s) used for DICOM export (if any).
- **`request`**: echoed request body.
- **`status`**: `done`.
- Exported file paths (per-stage directories `stage_0`, `stage_1`, ...):
  - **`nifti_prediction`**: list of paths to `prediction[_<suffix>].nii.gz`.
  - **`nifti_proba`**: list of paths to `proba[_<suffix>].nii.gz` if `save_proba_map=true`.
  - **`nifti_inputs`**: list of input NIfTI paths if `save_nifti_inputs=true`.
  - If `is_dicom=true`:
    - **`dicom_segmentation`**: list of paths to `prediction[_<suffix>].dcm`.
    - **`dicom_struct`**: list of paths to `struct[_<suffix>].dcm` if `save_rt_struct_output=true` and masks are non-empty.
    - **`dicom_fractional_segmentation`**: list of paths to fractional DICOM-SEG for probability maps if `save_proba_map=true`.
  - Empty predictions: when a stage’s mask is empty, DICOM-SEG/RTStruct export is skipped for that stage.

On failure:
- **HTTP 400** for invalid `nnunet_id` or invalid `series_folders` shape; payload includes `status="failed"` and `error` message.
- **HTTP 400** if `series_folders` is missing or inconsistent with the number of models.
- **HTTP 500** for runtime exceptions during inference; payload includes `status="failed"` and `error`.

### Examples

- **Discover models and schema**

```bash
curl -s http://localhost:12345/model_info | jq .
curl -s http://localhost:12345/request-params | jq .
```

- **Run single-model inference (NIfTI inputs)**

```bash
curl -X POST http://localhost:12345/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "nnunet_id": "prostate_whole_gland",
    "study_path": "/data/study01",
    "series_folders": ["inputs/seriesT2"],
    "output_dir": "/data/out/study01",
    "use_folds": [0,1,2,3,4],
    "tta": true,
    "save_proba_map": true,
    "proba_threshold": 0.1,
    "min_confidence": 0.5
  }'
```

- **Run multi-model cascade with DICOM input and RT Struct**

```bash
curl -X POST http://localhost:12345/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "nnunet_id": ["prostate_whole_gland", "prostate_zone"],
    "study_path": "/data/study02",
    "series_folders": [["inputs/seriesT2"], ["inputs/seriesT2"]],
    "output_dir": "/data/out/study02",
    "is_dicom": true,
    "cascade_mode": "intersect",
    "save_rt_struct_output": true
  }'
```

### Running as a Docker container

Firstly, users must install [Docker](https://www.docker.com/). **Docker requires `sudo` if not [correctly setup](https://docs.docker.com/engine/install/linux-postinstall/) so be mindful of this!**. Then:

1. Adapt the `model-serve-spec.yaml` with your favourite models; this is the blueprint for `model-serve-spec-docker.yaml` (same models but different model directory)
2. Build the container (`sudo docker build -f Dockerfile . -t nnunet_predict`)
3. Run the container while specifying the relevant ports (50422), GPU usage (`--gpus all`), and the model directory (`-v /models:/models`): `sudo docker run -it -p 50422:50422 --gpus all -v /models:/models nnunet_predict`. This will launch the inference server (described below)

### Operational notes

- **Strict GPU requirement:** The server requires an NVIDIA GPU and `nvidia-smi`. It waits for a GPU with at least the model’s `min_mem` free memory (`wait_for_gpu()`), using the maximum `min_mem` across models for multi-model requests. CPU-only systems are not supported.
- **CORS:** No CORS middleware is configured by default. If you expose the API to browsers, configure CORS as appropriate for your deployment.
- **Debug mode:** Set environment variable `DEBUG=1` to disable try/except around inference.
