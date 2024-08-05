import SimpleITK as sitk
import numpy as np
import os
import glob
import logging
import sys
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

        # Round label values to nearest integer
        arr_label = np.round(arr_label).astype(np.int32)

        # Create new label image with corrected orientation and metadata
        new_label = sitk.GetImageFromArray(arr_label)
        new_label.CopyInformation(image)

        return new_label
    except Exception as e:
        logger.error(f"Error in fix_label_headers: {str(e)}")
        return None

def process_folder(folder_path):
    try:
        # Find all .nrrd and .seg.nrrd files in the folder
        volume_files = glob.glob(os.path.join(folder_path, "*.nrrd"))
        label_files = glob.glob(os.path.join(folder_path, "*.seg.nrrd"))

        if not label_files:
            logger.warning(f"No label file found in {folder_path}")
            return

        label_file = label_files[0]  # Assume there's only one label file
        label = sitk.ReadImage(label_file, imageIO="NrrdImageIO")

        for volume_file in volume_files:
            if volume_file.endswith('.seg.nrrd'):
                continue  # Skip label files

            try:
                image = sitk.ReadImage(volume_file, imageIO="NrrdImageIO")
                aligned_label = fix_label_headers(image, label)

                if aligned_label is not None:
                    # Save the aligned label with the correct name
                    new_label_name = os.path.basename(volume_file).replace('.nrrd', '.seg.nrrd')
                    new_label_path = os.path.join(folder_path, new_label_name)
                    sitk.WriteImage(aligned_label, new_label_path)
                    logger.info(f"Matched and saved aligned label: {new_label_path}")
                    return  # Exit after finding a match
            except Exception as e:
                logger.error(f"Error processing {volume_file}: {str(e)}")

        logger.warning(f"No matching volume file found for label in {folder_path}")
    except Exception as e:
        logger.error(f"Error processing folder {folder_path}: {str(e)}")

def main(root_path):
    try:
        for folder_name in os.listdir(root_path):
            folder_path = os.path.join(root_path, folder_name)
            if os.path.isdir(folder_path):
                logger.info(f"Processing folder: {folder_path}")
                process_folder(folder_path)
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match and align NRRD files.")
    parser.add_argument("folder_path", help="Path to the folder containing NRRD files")
    args = parser.parse_args()

    root_path = args.folder_path
    logger.info(f"Starting processing for root path: {root_path}")
    main(root_path)