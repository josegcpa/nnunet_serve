# `nnunet_serve`

[![DOI](https://zenodo.org/badge/671509919.svg)](https://doi.org/10.5281/zenodo.17522202)

## Context

[nnUNet](https://github.com/MIC-DKFZ/nnUNet) is a relatively flexible framework. However, it is not exactly what people would call "production ready". With `nnunet_serve`, we have developed a container that allows users to run nnUNet as an API or as a CLI tool while keeping a relatively stable pool of models.

## Main features

1. Single case inference from and to multiple formats (from: Nifti, DICOM; to: Nifti, DICOM-seg, RT-struct, fractional DICOM-seg)
2. Batch inference using the aforementioned options (with background file writing to accelerate processing)
3. Model cascading: multiple models can be concatenated with being stuck to strict folder structures
    * Example 1: segment prostate → crop to prostate → detect prostate cancer
    * Example 2: segment prostate zones → crop to prostate zones → use prostate zones as input → segment csPCa
    * Example 3: segment liver → crop to liver → segment HCC → exclude HCC with 0% overlap with liver
4. Integration with Orthanc: Orthanc is the most open-source DICOM-web server, making nnunet_serve a very reasonable and appealing infrastructure for research
5. TotalSegmentator integration: TotalSegmentator is the largest suite of nnU-Net models for multiple CT and MRI tasks. We improve on their framework and greatly reduce inference times through refactoring and keeping series/inferences in memory
6. API: unlike typical workflows for nnU-Net, which depend on CLI-based routines, we have developed an API which guarantees integration with web-based services
7. Integration with both [SNOMED-CT](https://www.nlm.nih.gov/healthit/snomedct/index.html) and [EUCAIM](https://hyperontology.eucaim.cancerimage.eu/) ontologies: ontology integration allows the simple specification of DICOM-seg/RTstruct metadata, lifting the burden of generating custom files for specific structures

## Installation

Installation requirements are handled by `uv` (https://github.com/ultralytics/uv). `uv` is a tool for managing Python packages and dependencies.

### Requirements

* `uv` - using `uv` makes this all very easy as it manages Python packages. The installation is handled lazily (i.e. at runtime)
* CUDA-compatible GPU cards

## Operational notes

- **Strict GPU requirement:** The server requires an NVIDIA GPU and `nvidia-smi`. It waits for a GPU with at least the model’s `min_mem` free memory (`wait_for_gpu()`), using the maximum `min_mem` across models for multi-model requests. CPU-only systems are not supported.
- **CORS:** No CORS middleware is configured by default. If you expose the API to browsers, configure CORS as appropriate for your deployment.
- **Debug mode:** Set environment variable `DEBUG=1` to disable try/except around inference.

## Developer Documentation

The codebase follows **Google-style docstrings** for all functions and classes. If you are a developer looking to extend `nnunet_serve`, you can find detailed documentation for all core modules in the `src/nnunet_serve` directory.

## Citation

If you use this repository please cite the Zenodo repository as below.

**APA**

```
de Almeida, J. G., & Papanikolaou, N. (2026). josegcpa/nnunet_serve: v0.1.2 (v0.1.2). Zenodo. https://doi.org/10.5281/zenodo.17522203
```

**BibTex**

```
@software{de_almeida_2026_17522203,
  author       = {de Almeida, José Guilherme and
                  Papanikolaou, Nikolaos},
  title        = {josegcpa/nnunet\_serve: v0.1.2},
  month        = feb,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v0.1.2},
  doi          = {10.5281/zenodo.17522203},
  url          = {https://doi.org/10.5281/zenodo.17522203},
  swhid        = {swh:1:dir:af8aa6feda0eb9a33d98a4629a978bc289ad9537
                   ;origin=https://doi.org/10.5281/zenodo.17522202;vi
                   sit=swh:1:snp:ff077fba54804103b26419786f5f4035a9ae
                   3fa6;anchor=swh:1:rel:75c79771ab9c7e121eae8b4e50f4
                   5fe396abe1dc;path=josegcpa-nnunet\_serve-c5a1f06
                  },
}
```