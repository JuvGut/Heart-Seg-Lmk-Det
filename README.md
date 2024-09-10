# Heart-Valve-Segmentor and Landmark detection

## TO DO -> finish Segmentation & Landmark detection
- [ ] Visit Notion Kanban Board

# README:
### Description
This project aims to segment the heart into five segments (4 + Background) and detect landmarks in CT scans of hearts, using anonymized data provided by the [University Hospital of Basel](https://www.unispital-basel.ch/). The project utilizes [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) for segmentation and a custom 3D landmark detection pipeline based on the [Medical-Detection3d-Toolkit](https://github.com/qinliuliuqin/Medical-Detection3d-Toolkit/tree/master).

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

## Example output to the Segmentation model

The folder `Segmentation/Example_Output` contains 3 sample images with their ground truth labels, as well as the sample output of the segmentation model.

```
Segmentation/Example_Output/
├── Sample_images/
├── Sample_seg_labels/
└── Sample_seg_results/
```

## Project Structure
```
Project/
├── scripts/
│   ├── nrrd_processor.py
│   ├── change_label.py
│   ├── split_dataset.py
│   ├── create_dataset_json.py
│   └── ...
├── Documentation/
│   ├── Dataset_preparation.md (for seg and lmk)
│   ├── Landmark_detection_usage.md
│   ├── Segmentation_usage.md (TODO)
│   ├── nnunet-docker-guide.md
│   └── ...
├── nnUNet/
│   └── ...
├── landmark_detection/
│   └── ...
├── Segmentation/
│   └── Example_Output/
│       ├── Sample_images/
│       ├── Sample_seg_labels/
│       └── Sample_seg_results/
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

For detailed instructions, refer to [Dataset_preparation.md](/Documentation/Dataset_preparation.md).

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

For detailed instructions and additional options, refer to [how_to_use_nnunet.md](/nnUNet/documentation/how_to_use_nnunet.md)

The model can also be deployed using a docker container. In order to do that, consult this [guide](/nnUNet/documentation/nnunet-docker-guide.md).

### 3. Landmark Detection
Follow these steps to run the landmark detection pipeline:

a. Prealign the files, so the origin matches for all the files. 
```
python pre-align_images.py /path/to/dataset/folder
```

b. Convert .JSON landmarks to .CSV:
```
python convert_landmarks.py /path/to/landmarksTr/folder
```

c. Generate landmark masks:
```
python gen_landmark_mask.py -i <input_folder> -l <landmark_folder> -o <output_folder> -n <label_file> [-s <spacing>] [-b <bounds>]
```

d. Generate the dataset:
```
python gen_dataset.py
```

e. Train the landmark detection model:
```
python lmk_det_train.py
```

For detailed instructions and additional options, refer to [/Documentation/Landmark_detection_usage.md](Documentation/Landmark_detection_usage.md).

### Inference

a. prealign images
```
python pre-align_images.py /path/to/dataset/folder
```

b. generate test_file
```
python generate_test_file.py /path/to/test_images/folder
```

c. Run inference with the pretrained model

Adapt all the paths and run
```
python lmk_det_infer.py
```
## Authors and Acknowledgment
- Juval Gutknecht

## License
This project is licensed under the Center for medical Image Analysis and Navigation, Department of Biomedical Engineering, University of Basel.

## Contact
For any questions or issues, please open an issue on this repository or contact Juval Gutknecht.


# Appendix
## Findings
### Data
217 Folders containing the `.nrrd` volume file (sometimes there are several, i.e. once cropped, once not), the `.nrrd` segmentation file and one or several `.mrk.json` markup files.

- "Herz" Files: 103
- "Thorax/Abdomen" Files: 100
- Other Files: 13

Issues with 13 labels -> put whole contents of the BS-XXX folders into an `.ignore` folder:
- BS-339_11_Angio_Fl_Ao.Bo.Herz_axial___60%.nrrd
- BS-054_6_Herz__0.6__I30f__3__60%.nrrd
- BS-475_9_Herz__0.6__I26f__3__60%.nrrd
- BS-320_7_Herz__0.6__I30f__3__60%.nrrd
- BS-103_5_Fl_Thoracica__1.0__I26f__3__60%.nrrd
- BS-039_7_Herz___0.6__I26f__3__BestDiast_68_%.nrrd
- BS-582_8_Cor__Fl_Thx-Abd__0.6__I26f__3__60%.nrrd
- BS-255_8_Herz__1.0__Bv40__3__BestDiast_65_%.nrrd
- BS-053_12_Herz__0.6__I26f__3__70%.nrrd
- BS-055_404_PARENCHYME_1.5_iDose_(3).nrrd
- BS-469_8_Fl_Thoracica__1.0__I26f__3__60%.nrrd
- BS-061_6_Herz__0.6__I26f__3__BestDiast_68_%.nrrd
- BS-036_5_Fl_Herz__0.6__Bv40__3__65%.nrrd

**NEW Stats**
- "Herz" Files: 94
- "Thorax/Abdomen" Files: 97
- Other Files: 12

Now: Use all files that do not exit with an error to create dataset. Even the ones that do not have "Herz", "Thorax" or so in their name have the same segmentation masks.

This should result in a dataset consisting of 203 images and segmentations as well as landmarks BEFORE the splitting up of the data (80% Training, 20% Testing).

The dataset is then Validated by nnUNet and the datasets are prepared automatically (163 training images, rest for validation/testing).
