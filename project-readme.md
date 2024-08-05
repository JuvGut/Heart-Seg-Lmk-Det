# Heart-Valve-Segmentor and Landmark Detection

## Overview
This project aims to segment the heart into five segments (4 + Background) and detect landmarks in CT scans of hearts, using anonymized data provided by the University Hospital of Basel. The project utilizes [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) for segmentation and a custom 3D landmark detection pipeline based on the [Medical-Detection3d-Toolkit](https://github.com/qinliuliuqin/Medical-Detection3d-Toolkit/tree/master).

## Segments
- LVOT
- RCC
- LCC
- NCC
- BG (Background)

## Landmarks
- Commissure LCC-RCC
- Commissure LCC-NCC
- Commissure NCC-RCC
- Nadir LC
- Nadir RC
- Nadir NC
- Basis of IVT LCC-RCC
- Basis of IVT LCC-NCC
- Basis of IVT NCC-RCC

## Visuals
![alt text](BS-043.png)

## Project Structure
```
Project/
├── scripts/
│   ├── nrrd_processor.py
│   ├── change_label.py
│   ├── split_dataset.py
│   ├── create_dataset_json.py
│   └── ...
├── nnUNet/
│   └── ...
├── landmark_detection/
│   └── ...
└── README.md
```

## Installation
1. Clone this repository
2. Install the required dependencies (list them here or provide a requirements.txt file)

## Usage

### 1. Dataset Preparation
Follow these steps to prepare your dataset:

a. Match and align labels with images:
```
python scripts/dataset/nrrd_processor.py /path/to/folder/containing/data
```

b. Change the label structure (move background to 0):
```
python scripts/dataset/change_label.py /path/to/Dataset/labelsTr
```

c. Split the dataset into training and test sets:
```
python scripts/dataset/split_dataset.py /path/to/Dataset
```

d. Create the dataset.json file:
```
python scripts/dataset/create_dataset_json.py /path/to/Dataset
```

For detailed instructions, refer to `How-to-use-prepare-dataset.md`.

### 2. Segmentation with nnU-Net
Follow these steps to train and run inference with nnU-Net:

Set up the environment variables `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` according to [this document](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md). 

a. Experiment planning and preprocessing:
```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

b. Model training:
```
nnUNetv2_train DATASET_NAME_OR_ID CONFIGURATION FOLD [--npz]
```

c. Find the best configuration:
```
nnUNetv2_find_best_configuration DATASET_NAME_OR_ID -c CONFIGURATIONS
```

d. Run inference:
```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities
```

For detailed instructions and additional options, refer to `how_to_use_nnunet.md`.

### 3. Landmark Detection
Follow these steps to run the landmark detection pipeline:

a. Convert .JSON landmarks to .CSV:
```
python convert_landmarks.py /path/to/landmarksTr/folder
```

b. Generate landmark masks:
```
python gen_landmark_mask.py -i <input_folder> -l <landmark_folder> -o <output_folder> -n <label_file> [-s <spacing>] [-b <bounds>]
```

c. Generate the dataset:
```
python gen_dataset.py
```

d. Train the landmark detection model:
```
python lmk_det_train.py
```

For detailed instructions and additional options, refer to `A_How_to_use_this_pipeline.md`.

## Authors and Acknowledgment
- Juval Gutknecht

## License
This project is licensed under the Center for medical Image Analysis and Navigation, Department of Biomedical Engineering, University of Basel.

## Contact
For any questions or issues, please open an issue on this repository or contact Juval Gutknecht.
