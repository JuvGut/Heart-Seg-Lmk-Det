# Heart Valve Segmentation and Landmark detection

## README:
### Description
This project aims to segment the heart into 4 segments (4 + Background) and detect 9 landmarks in CT scans of hearts, using anonymized data provided by the [University Hospital of Basel](https://www.unispital-basel.ch/). The project utilizes [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) for segmentation and a custom 3D landmark detection pipeline based on the [Medical-Detection3d-Toolkit](https://github.com/qinliuliuqin/Medical-Detection3d-Toolkit/tree/master).

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

## Visuals of the provided labels and landmarks
![alt text](BS-043.png)

## Example output to the Segmentation model

The folder `Segmentation/Example_Output` contains 3 sample images with their ground truth labels, as well as the sample output of the segmentation model.

```
Project/
├── Segmentation/
│   └── Example_Output/
│       ├── Sample_images/
│       ├── Sample_landmarks/
│       └── Sample_lmk_results/
│       └── ...
├── Landmark_Detection/
│   └── Example_Output/
│       ├── Sample_images/
│       ├── Sample_seg_labels/
│       └── Sample_seg_results/
|       └── ...
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
│   ├── landmark-detection-docker-guide.md
│   └── ...
├── Medical-Detection3d-Toolkit-master/
│   ├── detection3d/
│       └── ...
│   ├── lmk_det_in_docker.sh
│   ├── lmk-pred-runner.sh
│   ├── run_detection.py
│   └── ...
├── Landmark_Detection/
│   └── Example_Output/
│       ├── Sample_images/
│       ├── Sample_seg_labels/
│       └── Sample_seg_results/
│       └── ...
├── nnUNet/
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
Follow these steps to **train** and **run inference** with nnU-Net:

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

d. Run inference (In order to run it in **Docker**, see below):
```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities
```

For detailed instructions and additional options, refer to [how_to_use_nnunet.md](/nnUNet/documentation/how_to_use_nnunet.md)

The model can also be deployed using a docker container. In order to do that, consult the [nnunet-docker-guide.md](/nnUNet/documentation/nnunet-docker-guide.md).

### 3. Landmark Detection
Follow these steps to run the landmark detection pipeline:

---
> Before all the following steps, navigate to the folder `/detection3d` within `Medical-Detection3d-Toolkit-master`.
---

a. Prealign the files, so the origin matches for all the files. 

> I achieved the best results, once the images (, segmentations) and landmarks had their origin at (0, 0, 0).

```
python scripts/prealign_images.py /path/to/dataset/folder
```

b. Convert .JSON landmarks to .CSV:
```
python scripts/convert_landmarks.py /path/to/landmarksTr/folder
```

c. Generate landmark masks:
```
python scripts/gen_landmark_mask.py -i <input_folder> -l <landmark_folder> -o <output_folder> -n <label_file> [-s <spacing>] [-b <bounds>]
```

d. Generate the dataset files:
> This step generates the files that contain the case names, the paths to the images, landmarks and landmark masks.
```
python gen_dataset.py
```

e. Train the landmark detection model:
> Specify all the paths and parameters in the config file (`/config/lmk_train_config.py`) and point to the config file from the file `detection3d/lmk_det_train.py`. Then start the training by running this command.
```
python lmk_det_train.py
```

For detailed instructions and additional options, refer to [Landmark_detection_usage.md](Documentation/Landmark_detection_usage.md).

### Inference

Run inference with a pretrained model.

Adapt all the paths in the file `lmk_det_infer.py` and run it like so. Alternatively it is possible to use flags to specify the paths. For that consult [Landmark_detection_usage.md](Documentation/Landmark_detection_usage.md) and for the usage of the model within Docker  [landmark-detection-docker-guide.md](Documentation/landmark-detection-docker-guide.md).

```
python lmk_det_infer.py
```

## Authors and Acknowledgment
- **Lead Developer:** Juval Gutknecht
- **Original nnUNet Implementation:** Isensee et al. ([GitHub Repository](https://github.com/MIC-DKFZ/nnUNet))
- **Original Medical Detection 3D Toolkit** Liu et al. ([GitHub Repository](https://github.com/qinliuliuqin/Medical-Detection3d-Toolkit))

## License
This project is licensed under the Center for medical Image Analysis and Navigation, Department of Biomedical Engineering, University of Basel.

## Contact
For general inquiries and bug reports, please open an issue in this repository
For specific questions, contact:
- Juval Gutknecht
- For questions about the original implementations, please refer to the respective repositories linked above.

# Appendix
## Preliminary findings of the provided data for this project.
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
