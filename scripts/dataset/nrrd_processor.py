import os
import shutil
import sys
import logging
from datetime import datetime
import SimpleITK as sitk
import numpy as np
import glob
import argparse

# Set up logging
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)  # Create the log folder if it doesn't exist

log_filename = f"nrrd_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(log_folder, log_filename)
logging.basicConfig(filename=log_filepath, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_label_headers(image, label):
    try:
        # Get numpy arrays
        arr_image = sitk.GetArrayFromImage(image)
        arr_label = sitk.GetArrayFromImage(label)

        # Check and correct orientation if necessary
        if arr_label.shape != arr_image.shape:
            logger.info(f"Original shapes - Image: {arr_image.shape}, Label: {arr_label.shape}")
            for i in range(3):
                for j in range(i+1, 3):
                    if arr_label.shape[i] == arr_image.shape[j] and arr_label.shape[j] == arr_image.shape[i]:
                        arr_label = np.swapaxes(arr_label, i, j)
                        logger.info(f"Swapped axes {i} and {j}")
                        break
                if arr_label.shape == arr_image.shape:
                    break
            logger.info(f"Final shapes - Image: {arr_image.shape}, Label: {arr_label.shape}")

        if arr_label.shape != arr_image.shape:
            logger.warning("Unable to match label shape to image shape")
            return None

        # Ensure we maintain the original data type
        original_dtype = label.GetPixelID()
        arr_label = arr_label.astype(sitk.GetArrayViewFromImage(label).dtype)

        # Create new label image with corrected orientation and metadata
        new_label = sitk.GetImageFromArray(arr_label)
        new_label.CopyInformation(image)

        # Ensure we use the original pixel type
        if new_label.GetPixelID() != original_dtype:
            new_label = sitk.Cast(new_label, original_dtype)

        return new_label
    except Exception as e:
        logger.error(f"Error in fix_label_headers: {str(e)}")
        return None

def copy_file(source_path, dest_path):
    try:
        shutil.copy2(source_path, dest_path)
        logger.info(f"Copied file: {source_path} -> {dest_path}")
    except Exception as e:
        logger.error(f"Error copying file {source_path}: {str(e)}")

def find_matching_files(patient_folder):
    files = [f for f in os.listdir(patient_folder) if not f.startswith('._')]
    
    # Find the segmentation file
    seg_file = next((f for f in files if f.endswith('.seg.nrrd')), None)
    
    if not seg_file:
        logger.warning(f"No segmentation file found in {patient_folder}")
        return None, None, None
    
    # Find the matching image file (preference for 'cropped' version)
    image_files = [f for f in files if f.endswith('.nrrd') and not f.endswith('.seg.nrrd')]
    cropped_image = next((f for f in image_files if 'cropped' in f.lower()), None)
    
    if cropped_image:
        image_file = cropped_image
    elif image_files:
        image_file = image_files[0]  # Take the first image file if no cropped version
    else:
        logger.warning(f"No matching image file found for {seg_file} in {patient_folder}")
        return None, None, None
    
    # Find landmark file
    landmark_file = next((f for f in files if f in ["MarkupsFiducial.mrk.json", "Landmarks.mrk.json"]), None)
    
    return image_file, seg_file, landmark_file

def process_and_organize_files(patient_folder, images_dir, labels_dir, landmarks_dir):
    try:
        image_file, seg_file, landmark_file = find_matching_files(patient_folder)
        
        if not image_file or not seg_file:
            logger.warning(f"Skipping folder {patient_folder} due to missing files")
            return
        
        # Process image file
        image_source_path = os.path.join(patient_folder, image_file)
        image_name = os.path.splitext(image_file)[0]
        image_name = image_name.replace(' ', '_')
        image_dest_path = os.path.join(images_dir, f"{os.path.basename(patient_folder)}_{image_name}_0000.nrrd")
        copy_file(image_source_path, image_dest_path)
        
        # Process segmentation file
        seg_source_path = os.path.join(patient_folder, seg_file)
        seg_dest_path = os.path.join(labels_dir, f"{os.path.basename(patient_folder)}_{image_name}.nrrd")
        
        # Fix label headers
        image = sitk.ReadImage(image_source_path, imageIO="NrrdImageIO")
        label = sitk.ReadImage(seg_source_path, imageIO="NrrdImageIO")
        aligned_label = fix_label_headers(image, label)
        
        if aligned_label is not None:
            writer = sitk.ImageFileWriter()
            writer.SetFileName(seg_dest_path)
            writer.SetUseCompression(True)
            writer.Execute(aligned_label)
            logger.info(f"Matched and saved aligned label: {seg_dest_path}")
        else:
            copy_file(seg_source_path, seg_dest_path)
        
        # Process landmark file if it exists
        if landmark_file:
            landmark_source_path = os.path.join(patient_folder, landmark_file)
            landmark_dest_path = os.path.join(landmarks_dir, f"{os.path.basename(patient_folder)}_{image_name}.mrk.json")
            copy_file(landmark_source_path, landmark_dest_path)
        else:
            logger.warning(f"No landmark file found in {patient_folder}")

    except Exception as e:
        logger.error(f"Error processing folder {patient_folder}: {str(e)}")

def process_main_folder(main_folder):
    try:
        dataset_dir = os.path.join(main_folder, "Dataset")
        images_dir = os.path.join(dataset_dir, "imagesTr")
        labels_dir = os.path.join(dataset_dir, "labelsTr")
        landmarks_dir = os.path.join(dataset_dir, "landmarksTr")

        for dir_path in [images_dir, labels_dir, landmarks_dir]:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")

        for item in os.listdir(main_folder):
            patient_folder = os.path.join(main_folder, item)
            if os.path.isdir(patient_folder) and item != "Dataset":
                logger.info(f"Processing folder: {patient_folder}")
                process_and_organize_files(patient_folder, images_dir, labels_dir, landmarks_dir)

    except Exception as e:
        logger.error(f"Error processing main folder {main_folder}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and organize NRRD files.")
    parser.add_argument("main_folder", help="Path to the main folder containing patient subfolders")
    args = parser.parse_args()

    main_folder = args.main_folder
    if not os.path.isdir(main_folder):
        print(f"Error: {main_folder} is not a valid directory.")
        logger.error(f"Invalid directory provided: {main_folder}")
        sys.exit(1)

    logger.info(f"Starting processing of main folder: {main_folder}")
    try:
        process_main_folder(main_folder)
        logger.info("Processing completed successfully.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during processing: {str(e)}")
    print(f"Processing complete. Log file: {log_filename}")