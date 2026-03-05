# API

This repository includes a FastAPI server that exposes nnU-Net inference as an HTTP API. The server is implemented in `src/nnunet_serve/nnunet_api.py` (with the application entrypoint in `src/nnunet_serve/nnunet_serve_api.py`) and configured by `model-serve-spec.yaml`.

## Model caching

Models are cached using a time-to-live cache system, they survive in memory for 5 minutes (300 seconds). Whenever a model is needed, it is checked if it is already cached. If it is not, it is loaded to the pre-specified cache and returned. The cache is cleaned up periodically (every 60 seconds) to free up space.

## Run the server

### Locally

```bash
# optionally set the port via env var (defaults to 12345)
export NNUNET_SERVE_PORT=12345

uv run uvicorn nnunet_serve.nnunet_serve_api:create_app \
  --host 0.0.0.0 \
  --port ${NNUNET_SERVE_PORT} \
  --reload
```

**Environment variables:**

- `MODEL_SERVE_SPEC`: path to a model serve spec file. Defaults to `model-serve-spec.yaml` in the working directory.
- `TOTALSEG_WEIGHTS_PATH`: optional override for where TotalSegmentator weights are downloaded/cached. Defaults to `<model_folder>/totalseg` based on `model-serve-spec.yaml`.
- `NNUNET_SERVE_PORT`: the port the server listens on (default: `12345`).

Ensure your `model-serve-spec.yaml` is present and correctly references your models. GPU and `nvidia-smi` must be available; the server waits for a GPU with enough free memory before running a job.

### Running as a Docker container

Firstly, users must install [Docker](https://www.docker.com/). **Docker requires `sudo` if not [correctly setup](https://docs.docker.com/engine/install/linux-postinstall/) so be mindful of this!**. Then:

1. Adapt the [`model-serve-spec.yaml`](model-serve-spec.yaml) with your favourite models; this is the blueprint for `model-serve-spec-docker.yaml` (same models but different model directory)
2. Build the container (`sudo docker build -f Dockerfile . -t nnunet_predict`)
3. Run the container while specifying the relevant ports (50422), GPU usage (`--gpus all`), and the model directory (`-v /models:/models`, as well as the output directory if necessary `-v /data/nnunet:/data/nnunet`): `docker run -it -p 50422:50422 --gpus all -v /models:/models -v /data/nnunet:/data/nnunet nnunet_predict uvicorn nnunet_serve.nnunet_serve_api:create_app`. This will launch the inference server. When specifying the output directory - if the outputs are not supposed to be kept, we recommend using a Docker volume which can be easily deleted. If the server is running internally, it might be interesting to mount a directory in the computer where outputs are stored.

## Endpoints

- **`GET /model_info`**
    - Returns the server’s model registry resolved from `model-serve-spec.yaml` and the filesystem.
    - Response model: `dict[str, Any]` (JSON object with model entries).

- **`GET /request-params`**
    - Returns the JSON schema of the request body for `/infer` (Pydantic model `InferenceRequest`).
    - Response model: `dict[str, Any]`.

- **`POST /infer`**
    - Runs inference for one or multiple models.
    - Response model: `InferenceResponse` (see response schema below).

- **`POST /infer_file`**
    - Accepts an archive upload (zip, tar, etc.), stores it, builds an `InferenceRequest`, and delegates to `/infer`. Keep in mind that while the `nnunet_serve` API does not require `study_path` for `/infer_file`, it still requires `series_folders`. This is to eliminate any ambiguity when selecting the relevant series for predictions.
    - Returns a job ID and inference result.
    - Response model: `dict[str, Any]` (includes job_id and same fields as `/infer`).

- **`GET /download/{job_id}`**
    - Serves the zip file containing the inference outputs for the given job ID.
    - Response class: `FileResponse` (application/zip).

- **`GET /healthz`**
    - Simple health check endpoint.
    - Response model: `dict[str, Any]` with `{"status": "ok"}`.

- **`GET /readyz`**
    - Readiness probe indicating whether models are loaded and a GPU is available.
    - Response model: `dict[str, Any]` with status and additional fields.

- **`GET /expire`**
    - Expires the TTL cache.
    - Response model: `dict[str, Any]` with status and message.

## Request body schema (InferenceRequest)

*please also refer to [nunet_serve.api_datamodels.InferenceRequest](../api/nunet_serve.api_datamodels.md#nunet_serve.api_datamodels.InferenceRequest)*

Required fields:

- **`nnunet_id`**: string or list of strings. Must match a model `id`, `name`, or any alias from `model-serve-spec.yaml`.
- **`study_path`**: string path to the study root directory (only for `/infer` endpoint; not necessary for `/infer_file`).
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
- `series_folders` also supports cascade references via `from:<model_or_alias>`, `from:<model_or_alias>=<label>`, and `from:<model_or_alias>[<index>]`. Missing upstream stages are injected automatically when needed.

## Response schema (POST /infer)

On success (HTTP 200):

- **`time_elapsed`**: seconds to complete the request.
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

## Response schema (POST /infer_file)

On success (HTTP 200):

- **`job_id`**: unique identifier for the inference job.
- All fields from the `/infer` response schema are included (`time_elapsed`, `nnunet_path`, `metadata`, `request`, `status`, exported file paths, etc.).
- The `request` field reflects the original request payload (without `study_path` as it is inferred from the uploaded file).

On failure (HTTP 400/500):

- Same error structure as `/infer` with an additional `job_id` field when applicable.
- Payload includes `status="failed"` and an `error` message describing the issue.

## Examples

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

