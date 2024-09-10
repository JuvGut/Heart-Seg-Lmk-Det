# How to use the Landmark detection algorithm

## Abstract

This 3d medical landmark detection pipeline ... 

## Step-By-Step

### 0. Step: Pre-Align the images.
Run the script with the dataset folder as an argument. Alternatively, change the default input folder in the file itself. 

*This script is located at heart-valve-segmentor/scripts/landmark_detection/pre-align_images.py*

```
python pre-align_images.py /path/to/Dataset/folder /path/to/output/folder
```

The script will set all the origins to (0, 0, 0), so all images are more or less aligned, instead of being scattered throughout space.
Process all the images, segmentations and landmarks. It starts with the Training images (Tr) and will automatically process the Test images (Ts) as well. It will then create the same folder structure in the output folder. 


### 1. Step: Convert .JSON landmarks to .CSV landmarks
Run the script with the input folder as an argument:

*This script is located at heart-valve-segmentor/Medical-Detection3d-Toolkit-master/detection3d/scripts/convert_landmarks.py*

```
python scripts/convert_landmarks.py /path/to/landmarksTr/folder
```

The script will:

Process all `.mrk.json` files in the input folder
Create a new folder named [input_folder_name]_csv in the parent directory
Convert each JSON file to a CSV file in the new folder

*Note: The script converts Slicer fiducial markup files (`.mrk.json`) to CSV format, extracting landmark names and coordinates.*

### 2. Step: Generate the landmark masks.
This mask is paramount for the model, because it shows the model where in the image the landmarks are found.
*This script is located at heart-valve-segmentor/Medical-Detection3d-Toolkit-master/detection3d/scripts/gen_landmark_mask.py*
How to Use gen_landmark_mask.py:

1. Place your input images (.nrrd files) in a folder.
2. Prepare a folder with landmark CSV files corresponding to your images.
3. Create a landmark label file (CSV) with columns 'landmark_name' and 'landmark_label'. Without that the code will fail.
4. Run the script from the command line:
```
python scripts/gen_landmark_mask.py -i <input_folder> -l <landmark_folder> -o <output_folder> -n <label_file> [-s <spacing>] [-b <bounds>]
```
Example:
```
python scripts/gen_landmark_mask.py -i /path/to/images -l /path/to/landmarks -o /path/to/output -n /path/to/label_file.csv
```
Alternatively, adapt the default inputs to your liking.

6. The program will generate landmark masks for each image and save them in the specified output folder.

Optional arguments (See below for explanation):
- -s: Specify spacing (default: [1.3, 0.85, 1.3])
- -b: Specify bounds (default: [3, 6])
- --debug: debugging mode with a verbose output
For more details, run `python gen_landmark_mask.py --help`.



### 3. Step: How to Use gen_dataset.py

1. Ensure your data is organized as follows:
   - Image files (.nrrd) in the `imagesTr` folder
   - Landmark CSV files in the `landmarksTr_csv` folder
   - Landmark mask files (.nii.gz) in the `landmark/mask` folder

2. Open `/scripts/gen_landmark_mask.py` and verify/update these paths at the bottom of the file:
   - `image_folder`
   - `landmark_file_folder`
   - `landmark_mask_folder`
   - `output_folder`

3. Run the script:
   ```
   python gen_dataset.py
   ```

4. Check the `output_folder` (gets created automatically, if doesn't exist yet) for the generated `train.csv` (and possibly `test.csv`) file(s).

Note: The script currently puts all images in the training set. Adjust the split ratio in the `split_dataset` function if you need a separate test set.

### 4. step: train the model -> How to Use Landmark Detection Training Script

1. Ensure your environment is set up with the required dependencies (PyTorch, tensorboardX, etc.).

2. Prepare your data and update the `config/lmk_train_config.py` file:
   - Set correct paths for `training_image_list_file` and `validation_image_list_file`.
   - Update `target_landmark_label` to match your landmarks.
   - Adjust `save_dir` to your desired output location.

3. Modify the FocalLoss parameters in `lmk_train_config.py`:
   - Update `focal_obj_alpha` to match the number of classes:
     ```python
     __C.landmark_loss.focal_obj_alpha = [0.75] * (len(__C.general.target_landmark_label) + 1)
     ```
     Note: Add 1 to account for the background class.

4. Adjust other parameters as needed:
   - `crop_size`, `sampling_size`, `num_pos_patches_per_image`, `num_neg_patches_per_image`, etc.

5. Run the training script:
   ```
   python lmk_det_train_.py -i /path/to/lmk_train_config.py -g 0
   ```
   Replace `/path/to/lmk_train_config.py` with the actual path to your config file.
   The `-g 0` argument specifies to use GPU 0. Adjust as needed.

6. Monitor the training:
   - Check the console output for loss values and other metrics.
   - Use TensorBoard to visualize training progress:
     ```
     tensorboard --logdir=/path/to/save_dir/tensorboard
     ```

7. After training, find your model checkpoints in the specified `save_dir`.

Note: If you encounter other errors, double-check all paths and ensure your data is formatted correctly.

The model is trained by running the program `lmk_det_train.py`. First all the parameters have to be adapted to the newly created dataset.

First the files `lmk_train_config.py` and `lmk_det_train.py` have to be adapted. This is done by opening each and adapting the parameters.

Then the model can be trained by running the following command.

``` 
python lmk_det_train.py
```

# Inference

## 1. Generate CSV from NRRD files

### Usage

```
python generate_test_file_big.py <input_folder> <output_file>
```

### Arguments

- `<input_folder>`: Path to the folder containing NRRD files.
- `<output_file>`: Path to the output CSV file. (This file doesn't have to exist, but only the folder.)

### Example

```
python generate_test_file_big.py /path/to/nrrd/files /path/to/output/test_file_big.csv
```

This script will:
1. Recursively search for all `.nrrd` files in the input folder.
2. Generate a CSV file with columns `image_name` and `image_path`.
3. If the output file already exists, it will append new data to it.

## 2. 3D Medical Image Segmentation Inference

THE MODEL WAS TRAINED USING THE MEAN SPACING PROVIDED BY THE PREPROCESSING OF THE NNUNETV2 PREPROCESSOR AND USED THROUGHOUT THE WHOLE LANDMARK DETECTION TRAINING.

### Usage

```
python lmk_det_infer_big.py [-i INPUT] [-m MODEL] [-o OUTPUT] [-g GPU_ID] [-s SAVE_PROB]
```
alternatively, use the bash script using the same arguments.
```
./prediction [-i INPUT] [-m MODEL] [-o OUTPUT] [-g GPU_ID] [-s SAVE_PROB]
```
### Arguments

- `-i, --input`: Input folder/file for intensity images (default: '/home/juval.gutknecht/Projects/CSA/DATA/big_test/dataset/test_file/test_file_big.csv')
- `-m, --model`: Model root folder (default: '/home/juval.gutknecht/Projects/CSA/DATA/results/model_big')
- `-o, --output`: Output folder for segmentation results (default: '/home/juval.gutknecht/Projects/CSA/DATA/big_test/inference_results')
- `-g, --gpu_id`: GPU ID to run the model (default: 5, set to -1 for CPU only)
- `-s, --save_prob`: Whether to save probability maps (default: False)

### Example

```
python lmk_det_infer_big.py -i /path/to/input/data -m /path/to/model -o /path/to/output/results -g 0 -s True
```

This script will:
1. Load the specified model.
2. Process the input images (single image, text file with image paths, or folder of images).
3. Perform 3D medical image segmentation.
4. Save the results in the specified output folder.

## Notes

- Ensure that the `detection3d` module is properly installed and accessible in your Python environment.
- Adjust the default paths in `lmk_det_infer_big.py` if necessary to match your system's directory structure.
- For large datasets, consider running the scripts on a machine with sufficient computational resources and GPU support.


# DEBUG

If some error occurs with the landmarks: check if all the landmark names are identical (Mostly *Basal RCC-NCC* instead of *Basis of IVT RCC-NCC* and *Nadir NC* instead of *Nadir NCS* for all three landmark types) and adapt it.d


# Additional Material

## Explanation of Parameters

### Spacing Parameter
The 'spacing' parameter defines the physical size of each voxel in the output landmark mask. It is specified as three float values representing the spacing in millimeters along the x, y, and z axes respectively.

- Default value: [0.43, 0.3, 0.43]
- Usage: -s 0.43 0.3 0.43 or --spacing 0.43 0.3 0.43

**How it affects the output:**

- The spacing parameter determines the resolution of the output landmark mask.
- A smaller spacing results in a higher resolution mask with more voxels, potentially capturing finer details but increasing computational requirements and file size.
- A larger spacing produces a lower resolution mask with fewer voxels, which may be computationally efficient but might lose some detail.
- The output mask is resampled to this spacing, regardless of the input image's original spacing.

### Bounds Parameter
The 'bounds' parameter consists of two values: the positive upper bound and the negative lower bound. These values define the size of the region around each landmark in the mask.

- Default values: [5, 10] (positive upper bound: 5, negative lower bound: 10)
- Usage: -b 5 10 or --bound 5 10

**How it influences landmark mask generation:**

1. Positive Upper Bound (first value):

- Defines the radius (in voxels) around the landmark where the mask will have the full landmark label value.
- Voxels within this radius from the landmark center are assigned the landmark's label value.


2. Negative Lower Bound (second value):

- Defines the outer radius (in voxels) of a "transition zone" around the landmark.
- Voxels between the positive upper bound and this value are assigned a value of 0.5, representing a transition or "negative" sample area.



**Visual representation:**
```
    0 0 0 0 0 0 0      0: Background
    0 0 5 5 5 0 0      5: Full landmark value (within positive upper bound)
    0 5 5 5 5 5 0      2: Transition zone (between bounds)
    0 5 5 L 5 5 0      L: Landmark center
    0 5 5 5 5 5 0
    0 0 5 5 5 0 0
    0 0 0 0 0 0 0
```
These bounds help create a gradual transition in the landmark mask, which can be beneficial for training landmark detection models by providing context around the exact landmark location.