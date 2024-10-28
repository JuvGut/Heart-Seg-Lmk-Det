# How to Prepare the Dataset for nnUNetv2 (and Landmark detection)

This guide provides detailed instructions on how to prepare your dataset for use with nnUNetv2. Follow the steps carefully to ensure your dataset is correctly structured and formatted.

## Step-by-Step Instructions


### Step 1: Match and align the label with the image and reorganize the Files into a Dataset.
Run `nrrd_processor.py` with the following command: 

```
python nrrd_processor.py /path/to/folder/containing/data
```

This program takes the label file and looks which available volume best matches the label. It then saves the label with the corresponding image filename like so: 

image: image_file.nrrd --> label: image_file.seg.nrrd

This step is necessary, because some folders contain multiple .nrrd volumes that don't correspond to the labels.

The folder containing all the data should include the CT-Scan files (.nrrd), the segmentation files (.seg.nrrd), and the landmark files (mrk.json).

The script reorganizes the files into a dataset with the following structure and appends _0000.nrrd to the volumes, which is required for nnUNet:

```
Dataset/
├── imagesTr/
├── labelsTr/
└── landmarksTr/
```


### Step 2: Change the Label Structure (Background to 0)

---
> Before running the script, create a **backup** of the labelsTr folder. It will prompt you to continue or cancel in the command line.
---

Run change_label.py with the following command:
```
python change_label.py /path/to/Dataset/labelsTr
```

This script rearranges the label files, specifically moving the "background" label from position 4 to position 0, as nnUNetv2 expects the background label to be at position 0.

### Step 3: Split dataset 
This script takes the imagesTr, labelsTr, and landmarksTr folders and generates the imagesTs, labelsTs, and landmarksTs folders, containing 20% of the total data as a test set. This ensures there is a test set of unseen data.

Run it with the following command:

```
python split_dataset.py /path/to/Dataset
```
The script assumes that your Dataset directory has the following structure before running:

```
Dataset/
├── imagesTr/
├── labelsTr/
├── landmarksTr/
```
After running the script, your Dataset directory will look like this:
```
CopyDataset/
├── imagesTr/
├── labelsTr/
├── landmarksTr/
├── imagesTs/
├── labelsTs/
├── landmarksTs/
```
with 20% of the data moved from the 'Tr' folders to the 'Ts' folders. This can be specified when changing the parameter `test_ratio` within the code.

### Step 4: Create 'dataset.json' file

Run the create_dataset_json.py script to generate the dataset.json file, which informs the model about the number of training examples, the modality of the input images, and the label structure. This file is needed to train the nnUNetv2.

```
python create_dataset_json.py /path/to/Dataset 
```

The generated dataset.json file will look similar to this:

```
{
        "name": "HeartScans",
        "description": "Heart segmentation",
        "tensorImageSize": "3D",
        "reference": "University Hospital Basel",
        "licence": "??????",
        "relase": f"1.0 16:54 12.07.2024}",
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0,
            "LVOT": 1,
            "RCC": 2,
            "LCC": 3,
            "NCC": 4
        },
        "numTraining": 0,
        "file_ending": ".nrrd"
    }
```
