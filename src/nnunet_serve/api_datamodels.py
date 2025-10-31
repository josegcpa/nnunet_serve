from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CascadeMode(Enum):
    INTERSECT = "intersect"
    CROP = "crop"


class CheckpointName(Enum):
    FINAL = "checkpoint_final.pth"
    BEST = "checkpoint_best.pth"


class Folds(Enum):
    ALL = "all"
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4


class InferenceRequestBase(BaseModel):
    """
    Data model for the inference request from local data. Supports providing
    multiple nnUNet model identifiers (``nnunet_id``) which in turn allows for
    intersection-based filtering of downstream results.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    nnunet_id: str | list[str] = Field(
        description="nnUnet model identifier or list of nnUNet model identifiers.",
        required=True,
    )
    output_dir: str = Field(
        description="Output directory.",
        required=True,
    )
    class_idx: int | list[int | None] | list[
        list[int] | int | None
    ] | None = Field(
        description="Prediction index or indices which are kept after each prediction",
        default=None,
    )
    checkpoint_name: CheckpointName | list[CheckpointName] = Field(
        description="nnUNet checkpoint name. Options are 'checkpoint_final.pth' and 'checkpoint_best.pth'.",
        default=CheckpointName.FINAL,
    )
    tmp_dir: str = Field(
        description="Directory for temporary outputs.", default=".tmp"
    )
    is_dicom: bool = Field(
        description="Whether series_paths refers to DICOM series folders.",
        default=False,
    )
    tta: bool = Field(
        description="Whether to apply test-time augmentation (use_mirroring)",
        default=False,
    )
    use_folds: list[Folds] = Field(
        description="Which folds should be used", default_factory=lambda: [0]
    )
    proba_threshold: float | list[float | None] | None = Field(
        description="(List of) probability threshold(s) for model output.",
        default=None,
    )
    min_confidence: float | list[float | None] | None = Field(
        description="(List of) minimum confidence(s) for model output.",
        default=None,
    )
    min_intersection: float | None = Field(
        description="Minimum intersection over the union to keep a candidate.",
        default=0.1,
    )
    crop_padding: tuple[int, int, int] | None = Field(
        description="Padding to be added to the cropped region.",
        default=(10, 10, 10),
    )
    cascade_mode: CascadeMode | list[CascadeMode] = Field(
        description="Whether to crop inputs to consecutive bounding boxes "
        "or to intersect consecutive outputs.",
        default=CascadeMode.INTERSECT,
    )
    save_proba_map: bool = Field(
        description="Saves the probability map", default=False
    )
    save_nifti_inputs: bool = Field(
        description="Saves the Nifti inputs in the output folder if input is DICOM",
        default=False,
    )
    save_rt_struct_output: bool = Field(
        description="Saves the output as an RT struct file", default=False
    )
    suffix: str | None = Field(
        description="Suffix for predictions", default=None
    )

    def __internal_update_field(self, field: str, value: Any):
        is_set = False
        if field in self.__pydantic_fields_set__:
            is_set = True
        self.__setattr__(field, value)
        if is_set is True:
            self.__pydantic_fields_set__.remove(field)

    def model_post_init(self, context):
        if self.save_proba_map and all(
            [x is None for x in self.proba_threshold]
        ):
            raise ValueError(
                "proba_threshold must be not-None if save_proba_map is True"
            )
        if isinstance(self.checkpoint_name, list) is False:
            self.checkpoint_name = [self.checkpoint_name]
        self.__internal_update_field(
            "checkpoint_name", [mode.value for mode in self.checkpoint_name]
        )
        if isinstance(self.cascade_mode, list) is False:
            self.cascade_mode = [self.cascade_mode]
        self.__internal_update_field(
            "cascade_mode", [mode.value for mode in self.cascade_mode]
        )


class InferenceRequest(InferenceRequestBase):
    study_path: str = Field(
        description="Path to study folder or list of paths to studies."
    )
    series_folders: list[str] | list[list[str]] = Field(
        description="Series folder names or list of series folder names (relative to study_path).",
        default=None,
    )
    crop_from: str | None = Field(
        description="Crops input to the bounding box of this mask.",
        default=None,
    )
    intersect_with: str | None = Field(
        description="Intersects output with this mask and if relative \
            intersection < min_intersection the object is deleted",
        default=None,
    )


class InferenceRequestFile(InferenceRequestBase):
    study_path: str | None = Field(
        description="Path to study folder or list of paths to studies.",
        default=None,
    )
    series_folders: list[str] | list[list[str]] | None = Field(
        description="Series folder names or list of series folder names (relative to study_path).",
        default=None,
    )


class InferenceResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    time_elapsed: float | None
    gpu: int | None
    nnunet_path: str | list[str] | None
    metadata: Any | None
    request: dict
    status: str
    error: str | None
