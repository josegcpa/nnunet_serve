# Docker container-ready nnUNet wrapper for SITK-readable and DICOM files

## Context

Given that [nnUNet](https://github.com/MIC-DKFZ/nnUNet) is a relatively flexible framework, we have developed a container that allows users to run nnUNet in a container while varying the necessary models. The main features are inferring all necessary parameters from the nnUNet files (spacing, extensions) and working for both DICOM folder and SITK-readable files. If the input is a DICOM, the segmentation is converted into a DICOM-seg file, compatible with PACS systems.

## Usage

### Standalone script

A considerable objective of this framework was its deployment as a standalone tool (for `bash`). To use it:

1. Install the necessary packages using an appropriate Python environment (i.e. `pip install -r requirements.txt`). We have tested this using Python `v3.11`
2. Run `python utils/entrypoint-prod.py --help` to see the available options
3. Segment away!

```bash
python utils/entrypoint-prod.py --help
```

```
usage: Entrypoint for nnUNet prediction. Handles all data format conversions. [-h] --series_paths SERIES_PATHS
                                                                              [SERIES_PATHS ...] --model_path MODEL_PATH
                                                                              [--checkpoint_name CHECKPOINT_NAME]
                                                                              --output_dir OUTPUT_DIR --metadata_path
                                                                              METADATA_PATH
                                                                              [--fractional_metadata_path FRACTIONAL_METADATA_PATH]
                                                                              [--empty_segment_metadata EMPTY_SEGMENT_METADATA]
                                                                              [--fractional_as_segments]
                                                                              [--study_uid STUDY_UID]
                                                                              [--folds FOLDS [FOLDS ...]] [--tta]
                                                                              [--tmp_dir TMP_DIR] [--is_dicom]
                                                                              [--proba_map]
                                                                              [--proba_threshold PROBA_THRESHOLD]
                                                                              [--min_confidence MIN_CONFIDENCE]
                                                                              [--rt_struct_output] [--save_nifti_inputs]
                                                                              [--intersect_with INTERSECT_WITH]
                                                                              [--min_intersection MIN_INTERSECTION]
                                                                              [--class_idx CLASS_IDX] [--suffix SUFFIX]
                                                                              [--job_id JOB_ID]
                                                                              [--update_url UPDATE_URL]
                                                                              [--success_message SUCCESS_MESSAGE]
                                                                              [--failure_message FAILURE_MESSAGE]
                                                                              [--log_file LOG_FILE] [--debug]

options:
    -h, --help            show this help message and exit
    --series_paths SERIES_PATHS [SERIES_PATHS ...], -i SERIES_PATHS [SERIES_PATHS ...]
                            Path to input series
    --model_path MODEL_PATH, -m MODEL_PATH
                            Path to nnUNet model folder
    --checkpoint_name CHECKPOINT_NAME, -ckpt CHECKPOINT_NAME
                            Checkpoint name for nnUNet
    --output_dir OUTPUT_DIR, -o OUTPUT_DIR
                            Path to output directory
    --metadata_path METADATA_PATH, -M METADATA_PATH
                            Path to metadata template for DICOM-Seg output
    --fractional_metadata_path FRACTIONAL_METADATA_PATH
                            Path to metadata template for fractional DICOM-Seg output (defaults to --metadata_path)
    --empty_segment_metadata EMPTY_SEGMENT_METADATA
                            Path to metadata template for when predictions are empty
    --fractional_as_segments
                            Converts the fractional output to a categorical DICOM-Seg with discretized probabilities (the
                            number of discretized probabilities is specified as the number of segmentAttributes in
                            metadata_path or fractional_metadata_path)
    --study_uid STUDY_UID, -s STUDY_UID
                            Study UID if series are SimpleITK-readable files
    --folds FOLDS [FOLDS ...], -f FOLDS [FOLDS ...]
                            Sets which folds should be used with nnUNet
    --tta, -t             Uses test-time augmentation during prediction
    --tmp_dir TMP_DIR     Temporary directory
    --is_dicom, -D        Assumes input is DICOM (and also converts to DICOM seg; prediction.dcm in output_dir)
    --proba_map, -p       Produces a Nifti format probability map (probabilities.nii.gz in output_dir)
    --proba_threshold PROBA_THRESHOLD
                            Sets probabilities in proba_map lower than proba_threhosld to 0
    --min_confidence MIN_CONFIDENCE
                            Removes objects whose max prob is smaller than min_confidence
    --rt_struct_output    Produces a DICOM RT Struct file (struct.dcm in output_dir)
    --save_nifti_inputs, -S
                            Moves Nifti inputs to output folder (volume_XXXX.nii.gz in output_dir)
    --intersect_with INTERSECT_WITH
                            Calculates the IoU with the sitk mask image in this path and uses this value to filter images
                            such that IoU < --min_intersection are ruled out.
    --min_intersection MIN_INTERSECTION
                            Minimum intersection over the union to keep a candidate.
    --class_idx CLASS_IDX
                            Class index.
    --suffix SUFFIX       Adds a suffix (_suffix) to the outputs if specified.
    --job_id JOB_ID       Job ID that will be used to post job status/create log file
    --update_url UPDATE_URL
                            URL to be used to post job status
    --success_message SUCCESS_MESSAGE
                            Message to be posted in case of success
    --failure_message FAILURE_MESSAGE
                            Message to be posted in case of failure
    --log_file LOG_FILE   Path to log file (with job_id, and success/failure messages)
    --debug               Enters debug mode
```

Example:

```bash
python utils/entrypoint-prod.py \
    -i study/series_1 study/series_2 study/series_3 \
    -o example_output/ \
    -m models/prostate_model \
    -M metadata_templates/metadata-template.json \
    -D -f 0 1 2 3 4 \
    --proba_map \
    --save_nifti_inputs
```

### Running as a Docker container

Firstly, users must install [Docker](https://www.docker.com/). **Docker requires `sudo` if not [correctly setup](https://docs.docker.com/engine/install/linux-postinstall/) so be mindful of this!**. Then:

1. Build the container (`sudo docker build -f Dockerfile . -t nnunet_predict`)
2. Run the container. We have replicated this as an additional script (`utils/entrypoint-with-docker.py`) with the same arguments as those specified to run as a standalone tool with the addition of a `-c` flag specifying the name of the Docker image.

With `utils/entrypoint-with-docker.py`, this:

```
docker run \
    --gpus all \
    --user "$(id -u):$(id -g)" \
    -v $(dirname $(realpath $INPUT_PATHS)):/data/input \
    -v $(realpath $OUTPUT_FOLDER):/data/output \
    -v $(realpath $MODEL_FOLDER):/model \
    -v $(dirname $(realpath $METADATA_TEMPLATE)):/metadata \
    --rm \
    $DOCKER_IMAGE \
    -i $file_names_in_docker -d -M $metadata_name_in_docker
```

becomes this (for a DICOM input):

```
python utils/entrypoint-with-docker.py \
    -i $INPUT_PATHS \
    -o $OUTPUT_FOLDER \
    -m $MODEL_FOLDER \
    -d \
    -M $METADATA_TEMPLATE \
    -c $DOCKER_IMAGE
```

### Logging and status updates

To facilitate integration into production environments, we have added a logging function to `entrypoint-prod.py`. This works by specifying the following CLI arguments:

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
