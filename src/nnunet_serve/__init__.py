import os
import warnings

IGNORE_WARNINGS = [
    "Please import `label` from the `scipy.ndimage` namespace;",
    "Please import `grey_dilation` from the `scipy.ndimage` namespace;",
    "Please import `gaussian_gradient_magnitude` from the `scipy.ndimage` namespace;",
    "Please import `gaussian_filter` from the `scipy.ndimage` namespace;",
    "DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute",
]
for warn in IGNORE_WARNINGS:
    warnings.filterwarnings("ignore", message=warn)


for k in ["nnUNet_preprocessed", "nnUNet_raw", "nnUNet_results"]:
    dir = "tmp" if k != "nnUNet_preprocessed" else "tmp/preproc"
    os.environ[k] = os.environ.get(k, dir)
