# How to use the Landmark detection algorithm

## Abstract

This 3d medical landmark detection pipeline ... 

## Step-By-Step
### Initial Setup
Navigate to the detection toolkit directory: 
```
cd Medical-Detection3d-toolkit-master/detection3d
```

### 0. Step: Pre-Align images
This initial preprocessing step ensures all images share a common origin point (0, 0, 0), which significantly improved landmark detection accuracy during training.
Run the alignment script:

```
python scripts/pre-align_images.py /path/to/Dataset/folder /path/to/output/folder
```

The script will:
- Process all the images, segmentations and landmarks
- Handle both Training (Tr) and Test (Ts) datasets automatically
- Maintain the original folder structure in the output directory 
- Set all image origins to (0, 0, 0), preventing spatial scatter

### 1. Step: Convert .JSON landmarks to .CSV landmarks
Run the script with the input folder as an argument:

```
python scripts/convert_landmarks.py /path/to/landmarksTr/folder
```

The script will:

- Process all `.mrk.json` files in the input folder
- Create a new folder named [input_folder_name]_csv in the parent directory
- Convert each JSON file to a CSV file in the new folder

*Note: The script converts Slicer fiducial markup files (`.mrk.json`) to CSV format, extracting landmark names and coordinates. It will **not** retain all the additional information contained in the Slicer fiducial markup files.*

### 2. Step: Generate the landmark masks.
This step creates mask files that indicate the precise locations of the anatomical landmarks in each image. These masks are essential for training the model to recognize landmark positions.

**Prerequisites**
- Prealigned images in `imagesTr` folder
- Landmark files in .csv format (converted in Step 1)
- Landmark label file (.csv) containing: 
   - Column 'landmark_name': Name of each landmark
   - Column 'landmark_label': Numeric label for each landmark

**Basic Usage**
```
python scripts/gen_landmark_mask.py \ 
   -i /path/to/imagesTr \ 
   -l /path/to/landmarksTr_csv \
   -o path/to/landmark_masks \
   -n /path/to/label_file.csv
```

Alternatively, adapt the default inputs to your liking.

**How It Works**
The script:
- Matches each image with its corresponding landmark file
- Processes landmarks using the image's native spacing
- Creates a mask where:
   - Positive values indicate landmark locations (using labels from label file)
   - -1 indicates the negative boundary region
   - 0 indicates background
- Saves masks as .nii.gz files in the output folder
**Output**
Generated masks will be named: `[original_image_name]_landmark_mask.nii.gz`

### 3. Step: Generate Dataset Split
This step organizes the data into training and test sets by creating .csv files that link images with their corresponding landmarks and masks.

**Prerequisites**
Ensure you have:
- Pre-aligned images (from Step 0)
- Converted landmark files (from Step 1)
- Generated landmark masks (from Step 2)

**Basic Usage**
```
python scripts/generate_dataset.py
```

**How it works**
The script: 
- Randomly splits the dataset (80% training, 20% test/validation)
- Creates two .csv files (`train.csv` and `test.csv`) containing: 
   - Image names
   - Full paths to images
   - Full paths to landmark files
   - Full paths to landmark masks

**Important Notes**
- Uses a fixed random seed (0) for reproducible splits
- Automatically creates the output folder if it doesn't exist
- Files are matched based on the first 6 characters of the filename (e.g., "BS-001")
- Only processes `.nrrd` image files
- Missing landmark files or masks will be recorded as empty paths in the CSV

### 4. step: Train the Landmark Detection Model
This step covers how to configure and train the landmark detection model on the preprocessed dataset. 

**Prerequisites**
- Completed Steps 0-3 (aligned images, converted landmarks, generated masks, created dataset split)
- Environment with required packages:

   `PyTorch tensorboardX SimpleITK pandas numpy`

**Configuration Setup**

Update config/lmk_train_config.py
```
# Dataset paths
__C.general.training_image_list_file = '/path/to/Dataset012_aligned/train.csv'
__C.general.validation_image_list_file = '/path/to/Dataset012_aligned/test.csv'
__C.general.save_dir = '/path/to/output/landmark_detection_model'

# Landmark configuration
__C.general.target_landmark_label = {
    'landmark1': 1,
    'landmark2': 2,
    # Add all your landmarks here
}

# Loss function parameters
__C.landmark_loss.focal_obj_alpha = [0.75] * (num_landmarks + 1)  # +1 for background

# Training parameters
__C.train.crop_size = [96, 96, 96]      # Size of training patches
__C.train.batch_size = 4                # Adjust based on GPU memory
__C.train.num_pos_patches_per_image = 8  # Positive sample patches per image
__C.train.num_neg_patches_per_image = 8  # Negative sample patches per image
```

**Training Process**

1. Start training:
```
python lmk_det_train.py -i config/lmk_train_config.py -gpu_id 0
```
Arguments:
- -i: Path to configuration file
- -g: GPU ID (default: 0)

Or specify these as default values in the file `lmk_det_train.py`.

2. Monitor Training:

```
tensorboard --logdir=/path/to/output/landmark_detection_model
````

Note: If you encounter other errors, double-check all paths and ensure your data is formatted correctly.

# Inference
This section explains how to use the trained model to detect landmarks in new images.

### Option A: Direct Python Inference (local)

**Basic Usage**
```
python lmk_det_infer.py -i <input> -m <model_path> -o <output_dir> -g <gpu_id>

```
Example:
```
python lmk_det_infer.py \
    -i /path/to/test/images \
    -m /path/to/model/ \
    -o /path/to/results \
    -g 0
```

Input Options
- Single image: Path to an CT image file
- Multiple images:
   - Folder containing the CT image files
   - Text file with image paths (one per line)

Parameters
- `-i`: Input path (image/folder/list)
- `-m`: Path to trained model checkpoint
- `-o`: Output directory for results
- `-g`: GPU ID (default: 0)

Output
- CSV files containing detected landmark coordinates

### Option B: Docker Inference
Refer to detailed Docker guide in [Documentation/landmark-detection-doocker-guide.md](Documentation/landmark-detection-docker-guide.md) for containerized inference.

## Evaluation of Detection Model Performance

This step analyzes detection accuracy and generates comprehensive reports comparing ground truth and detected landmarks.

**Prerequisites**
- Pre-aligned images (from Step 0)
- Ground truth landmark files (from Step 1)
- Inference results from the trained model
- Python packages: `numpy pandas matplotlib SimpleITK``

**Basic Usage**

```
python scripts/gen_html_report.py \
    --image_folder /path/to/test/images \
    --label_folder /path/to/ground_truth/landmarks \
    --detection_folder /path/to/model/predictions \
    --output_folder /path/to/report/output \
    --generate_pictures  # Optional: creates visualization images
```

**Generated Reports**
The script produces a comprehensive analysis including:

1. **HTML Report**

- Overall model performance metrics
- Per-landmark statistics
- Interactive visualizations
- Case-by-case analysis

2. **CSV Reports**

- landmark_summary.csv: Statistics for each landmark
- case_summary.csv: Per-case detection rates
- detailed_errors.csv: Comprehensive error analysis

3. **Visualizations**

- Error distributions
- Axis-specific analyses
- Resolution impact analysis
- Detection rate distributions
- Top 10 challenging landmarks

4. **Detailed Analysis**

- Mean, median, and standard deviation of errors
- Resolution impact assessment
- Outlier analysis
- Axis-specific performance metrics

### Key Metrics Explained

- Detection Rate: Percentage of successfully detected landmarks
Mean Error: Average distance between predicted and true positions (mm)
- Axis-specific Errors: Separate analysis for X, Y, and Z coordinates
- 5mm Inlier Rate: Percentage of detections within 5mm of ground truth
- Resolution Impact: Correlation between image resolution and detection accuracy

**Tips**

- Use --debug flag for detailed processing information
- Enable --generate_pictures for visual validation
- Review the detailed_analysis.md file for in-depth insights
- Check error distributions to identify systematic issues

