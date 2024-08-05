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


# DEBUG

If some error occurs with the landmarks: check if all the landmark names are identical (i.e. Basis of RCC-NCC vs Basis of IVT RCC-NCC) and adapt it.
