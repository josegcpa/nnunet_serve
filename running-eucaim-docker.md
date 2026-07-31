# Instructions

1. Make sure that you have Docker installed and a working GPU card
2. Identify the study path, the series folder inside the study path and the output directory. In my case: 
   - `-i` (`--study_path`) is `/home/jose_almeida/projects/nnunet_docker/example/example_real/t2axial_dicoms`
   - `-s` (`--series_folders`) is `MR_T2W_TSE_ax`
   - `-o` (`--output_dir`) is `/home/jose_almeida/projects/nnunet_docker/tmp_output`
3. Make sure that the output directory is writable by anyone! This is very important. In UNIX, run `sudo chmod 777 tmp_output/` or `chmod 777 tmp_output/`
4. Run the Docker command:
```
docker run \
    -v /home/jose_almeida/projects/nnunet_docker/example/example_real/t2axial_dicoms/2015050934_RM_PELVICA:/data/input \
    -v /home/jose_almeida/projects/nnunet_docker/tmp_output:/data/output \
    --gpus all \
    harbor.eucaim.cancerimage.eu/processing-tools/champ-prostate-zone-segmentation nnunet-predict \
    -i /data/input \
    -s MR_T2W_TSE_ax \
    -o /data/output \
    --is_dicom
```