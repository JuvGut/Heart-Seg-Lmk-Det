# How to use the Landmark detection algorithm

## Abstract

This 3d medical landmark detection pipeline ... 

## Step-By-Step
 
### 1. Step: Convert .JSON landmarks to .CSV landmarks
Run the script with the input folder as an argument:

```
python convert_landmarks.py /path/to/landmarksTr/folder
```

The script will:

Process all `.mrk.json` files in the input folder
Create a new folder named [input_folder_name]_csv in the parent directory
Convert each JSON file to a CSV file in the new folder

*Note: The script converts Slicer fiducial markup files (`.mrk.json`) to CSV format, extracting landmark names and coordinates.*

### 2. Step: Generate the landmark masks.
How to Use gen_landmark_mask.py:

1. Place your input images (.nrrd files) in a folder.
2. Prepare a folder with landmark CSV files corresponding to your images.
3. Create a landmark label file (CSV) with columns 'landmark_name' and 'landmark_label'.
4. Run the script from the command line:
```
python gen_landmark_mask.py -i <input_folder> -l <landmark_folder> -o <output_folder> -n <label_file> [-s <spacing>] [-b <bounds>]
```
Example:
```
python gen_landmark_mask.py -i /path/to/images -l /path/to/landmarks -o /path/to/output -n /path/to/label_file.csv
```

6. The program will generate landmark masks for each image and save them in the specified output folder.

Optional arguments:
- -s: Specify spacing (default: [1.3, 0.85, 1.3])
- -b: Specify bounds (default: [3, 6])

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

### Usage

```
python lmk_det_infer_big.py [-i INPUT] [-m MODEL] [-o OUTPUT] [-g GPU_ID] [-s SAVE_PROB]
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
