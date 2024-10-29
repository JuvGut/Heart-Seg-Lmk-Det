# Project Modifications Documentation

Copyright (C) 2024 Center for medical Image Analysis and Navigation, 
Department of Biomedical Engineering, University of Basel

This documentation is part of a project licensed under GNU General Public License v3.0.
For full license terms, see the LICENSE file in the root directory.

## Source Repositories
1. nnUNet
   - Original Repository: [nnUNet](https://github.com/MIC-DKFZ/nnUNet)
   - Version Used: v2.5.1
   - Status: Unchanged, used as-is
   - License: [Apache License 2.0](https://github.com/MIC-DKFZ/nnUNet/blob/master/LICENSE)

2. Medical-Detection3d-Toolbox
   - Original Repository: [Medical-Detection3d-Toolbox](https://github.com/qinliuliuqin/Medical-Detection3d-Toolkit/tree/master)
   - Version Used: prior to March 2022
   - Status: Significantly modified
   - License: [GNU General Public License v3.0](https://github.com/qinliuliuqin/Medical-Detection3d-Toolkit/blob/master/LICENSE)

## Medical-Detection3d-Toolbox Modifications
While exact changes cannot be tracked due to historical workflow, the following major modifications were made:

### Structural Changes
- File added: detection3d/config/lmk_train_config_1.py
- File added: detection3d/config/lmk_train_config_aligned.py
- File added: detection3d/scripts/convert_landmarks.py
- File added: detection3d/scripts/generate_test_file.py
- File added: detection3d/scripts/prealign_images.py
- File added: detection3d/utils/resample_resolution.py
- File added: detection3d/utils/resampler.py
- File added: detection3d/lmk_det_train_1.py
- Not necessarily removed any files, some simply didn't exist at the time.

### Functional Changes
- Changes in landmarks to be processed
- Addition of .nrrd processing capabilities

### Modified Components
- `detection3d/config/lmk_train_config.py`, `detection3d/config/lmk_train_config_1.py`, `detection3d/config/lmk_train_config_aligned.py`
    - Purpose of changes: Adaption of config file(s) to the Dataset at hand
    - Major modifications: Changes in the parameters
- `detection3d/core/lmk_det_infer.py`
    - Purpose of changes: Addition of .nrrd reading capabilities, Lowering detection threshold 
    - Major modifications: Lowering detection threshold from 0.5 to 0.25
- `detection3d/scripts/convert_landmarks.py`
    - Purpose of changes: Convert Landmarks from .mrk.json to .csv to comply with pipeline requirements
    - Major modifications: Addition of the script altogether.
- `detection3d/scripts/generate_test_file.py`
    - Purpose of changes: Script generates a file with paths to test images.
    - Major modifications: Addition of the script altogether.
- `detection3d/scripts/prealign_images.py`
    - Purpose of changes: This script sets all the image origins to (0, 0, 0)
    - Major modifications: Addition of the script altogether.
- `detection3d/scripts/resample_resolution.py`
    - Purpose of changes: This script sets all the image, labels, and landmarks origins to (0, 0, 0) and it resamples them to a specified shape. Not used in the final version.
    - Major modifications: Addition of the script altogether.
- `detection3d/scripts/resampler.py`
    - Purpose of changes: This script sets all the image origins to (0, 0, 0) and it resamples the images to a specified spacing. Not used in the final version.
    - Major modifications: Addition of the script altogether.
- `detection3d/lmk_det_train.py`, `detection3d/lmk_det_train_1.py`
    - Purpose of changes: Changes to Parameters for the model to work in the pipeline.
    - Major modifications: Automatically changes the working directory, so the model can be called from anywhere.

## Integration
I combined two repositories, both specializing in a similar but different discipline - nnUNet in segmentation tasks, Medical-Detection3d-Toolbox in medical landmark detection. 
Both used the same dataset and dataset structure, this is why there is a folder `scripts/dataset` in this repository. To generate or prepare a dataset for this repository, one can follow the [Documentation](heart-valve-segmentor/Documentation/Dataset_preparation.md). 

The nnUNet remained mostly unchanged and was used to train a model. Afterwards, work was done containerizing the [inference](heart-valve-segmentor/Segmentation). To use this Docker containerization, one can follow the guide [nnUNet Docker Guide](heart-valve-segmentor/Documentation/nnunet-docker-guide.md).

The pipeline of the repository Medical-Detection3d-Toolbox was adapted to handle .nrrd and the also the structure of the dataset of the model nnUNet. To follow the preparation and usage refer to the documentation [Landmark detection usage](heart-valve-segmentor/Documentation/Landmark_detection_usage.md) and [Landmark detection docker guide](heart-valve-segmentor/Documentation/landmark-detection-docker-guide.md).
