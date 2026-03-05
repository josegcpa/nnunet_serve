# Command line interface

The command-line interface (CLI) is a powerful tool that allows you to run nnUNet predictions directly from the command line. This is particularly useful for batch processing and integration into larger workflows.

## Single case inference

To use the CLI:

1. Run `uv run nnunet-predict --help` to see the available options
2. Segment away!

```bash
uv run nnunet-predict --help
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
  --use_folds, -f FOLDS [FOLDS ...]
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
uv run nnunet-predict \
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

Example with `from:` references in the same stage input list:

```bash
uv run nnunet-predict \
  --study_path path/to/study \
  --series_folders seriesT2,seriesDWI,seriesADC,from:prostate_zone_mri=1,from:prostate_zone_mri=2 \
  --nnunet_id prostate_clinically_significant_lesion_bpmri \
  --output_dir path/to/output \
  --is_dicom \
  --cascade_mode crop \
  --proba_map \
  --proba_threshold 0.1
```

### Extended argument description

A core concept underlies this framework - that of cascading predictions. The output of a prediction is used as an input for the next prediction by either cropping the input image, filtering objects in the output image based on a minimum intersection or by appending it to the input image. Fields flagged with 💧 support multiple values in compliance with the cascade. For these fields, multiple space-separated values can be specified as long as the number of values matches the number of models (`nnunet_id`) in the cascade. In some instances, multiple values at each stage might require specification (`series_folders` or `folds`). For these, at each stage, multiple values can be specified using commas (`,`).

- **`--study_path` / `-i`**: Path to the input study directory containing the imaging data. Required.
- **`--series_folders` / `-s`**: One or more relative paths to series folders within the study. Required. Multiple **space separated** values refer to multiple stages of the cascade. At each stage, different series can be specified using commas (`,`) 💧
- **`--series_folders` / `-s` advanced (`from:` syntax)**: In cascades, you can reference a prior stage prediction as an input channel using `from:<model_or_alias>`. Optional selectors are supported:
  - `from:<model_or_alias>` → full predicted mask (`prediction.nii.gz`)
  - `from:<model_or_alias>=<label>` → binary mask for one label (for example `=1`)
  - `from:<model_or_alias>[<index>]` → indexed volume/channel access
  This allows "late" models to consume outputs from earlier models without manually creating intermediate files.
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


## CLI with status updates
*for `uv run python -m nnunet_serve.entrypoints.entrypoint_prod`*

As an example of an integration into a production environment, the `entrypoint_prod.py` script includes a logging function that works through a REST API. This works by specifying the following CLI arguments:

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

## Batch inference

The `entrypoint_batch.py` script enables running inference on multiple studies defined in a JSON file.

### Data JSON format

Create a JSON file (e.g., `data_json.json`) containing a list of dictionaries, each with the keys:

- `study_path`: path to the study directory.
- `series_folders`: list of series folder lists (matching the cascade format).
- `output_dir`: directory where outputs for that study will be written.

Example (`data_json.json`):

```json
[
    {
        "study_path": "example",
        "series_folders": [["dcm"]],
        "output_dir": "test_output/entrypoint_output_batch"
    },
    {
        "study_path": "example_2",
        "series_folders": [["dcm"]],
        "output_dir": "test_output/entrypoint_output_batch_2"
    }
]
```

### Data directory format

The data directory format is an alternative to the data JSON format - it probably easier for centers which follow a minimally structured data organization with patient/study/series format and where each series is tagged with an underscore-separated indicator similar to nnU‑Net (e.g. 'series_0000', 'series_0001', etc.).

This can be used as follows:

- `--data_dir`: Path to a hierarchical directory containing patient/study/series folders. Each series folder must be named with an underscore‑separated index (e.g., `series_0000`, `series_0001`, …). This option is mutually exclusive with `--data_json` and requires `--output_dir` to specify where results will be written.

### Running batch inference

Using the dataset JSON:

```bash
uv run nnunet-predict-batch \
    --data_json data_json.json \
    --nnunet_id prostate prostate_zones \
    --use_folds 0 1 2 3 4 \
    --tta \
    --proba_map \
    --proba_threshold 0.1 \
    --min_confidence 0.5 \
    --cascade_mode crop \
    --save_nifti_inputs
```

Using the data directory (requires specifying `--output_dir` as well):

```bash
uv run nnunet-predict-batch \
    --data_dir <data_dir> \
    --output_dir <output_dir> \
    --nnunet_id prostate prostate_zones \
    --use_folds 0 1 2 3 4 \
    --tta \
    --proba_map \
    --proba_threshold 0.1 \
    --min_confidence 0.5 \
    --cascade_mode crop \
    --save_nifti_inputs
```

All CLI arguments supported by `nnunet-predict` are available; the script forwards them to each study entry. Either `--data_json` or `--data_dir` (with `--output_dir`) must be provided for batch mode.

Refer to `src/nnunet_serve/entrypoints/entrypoint_batch.py` for the full implementation.

