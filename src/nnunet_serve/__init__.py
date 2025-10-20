import os

for k in ["nnUNet_preprocessed", "nnUNet_raw", "nnUNet_results"]:
    dir = "tmp" if k != "nnUNet_preprocessed" else "tmp/preproc"
    os.environ[k] = os.environ.get(k, dir)
