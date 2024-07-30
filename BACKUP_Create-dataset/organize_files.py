import os
import shutil
import sys
import logging
from datetime import datetime

# Set up logging
log_filename = f"file_organizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def copy_file(source_path, dest_path):
    try:
        shutil.copy2(source_path, dest_path)
        logging.info(f"Copied file: {source_path} -> {dest_path}")
    except Exception as e:
        logging.error(f"Error copying file {source_path}: {str(e)}")

def organize_files(patient_folder, images_dir, labels_dir, landmarks_dir):
    try:
        files = [f for f in os.listdir(patient_folder) if not f.startswith('._')]

        volume_file = next((f for f in files if f.endswith('.nrrd') and not f.endswith('.seg.nrrd')), None)
        if volume_file:
            image_name = os.path.splitext(volume_file)[0]
            source_path = os.path.join(patient_folder, volume_file)
            dest_path = os.path.join(images_dir, f"{os.path.basename(patient_folder)}_{image_name}_0000.nrrd")
            copy_file(source_path, dest_path)

            # Look for matching segmentation file
            seg_file = f"{image_name}.seg.nrrd"
            if seg_file in files:
                source_path = os.path.join(patient_folder, seg_file)
                dest_path = os.path.join(labels_dir, f"{os.path.basename(patient_folder)}_{seg_file}")
                copy_file(source_path, dest_path)
            else:
                logging.warning(f"No matching segmentation file found for {volume_file} in {patient_folder}")
        else:
            logging.warning(f"No volume file found in {patient_folder}")

        landmark_file = next((f for f in files if f in ["MarkupsFiducial.mrk.json", "Landmarks.mrk.json"]), None)
        if landmark_file:
            source_path = os.path.join(patient_folder, landmark_file)
            dest_path = os.path.join(landmarks_dir, f"{os.path.basename(patient_folder)}_{os.path.basename(image_name)}.mrk.json")
            copy_file(source_path, dest_path)
        else:
            logging.warning(f"No landmark file found in {patient_folder}")

    except Exception as e:
        logging.error(f"Error processing folder {patient_folder}: {str(e)}")

def process_main_folder(main_folder):
    try:
        dataset_dir = os.path.join(main_folder, "Dataset")
        images_dir = os.path.join(dataset_dir, "imagesTr")
        labels_dir = os.path.join(dataset_dir, "labelsTr")
        landmarks_dir = os.path.join(dataset_dir, "landmarksTr")

        for dir_path in [images_dir, labels_dir, landmarks_dir]:
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Created directory: {dir_path}")

        for item in os.listdir(main_folder):
            patient_folder = os.path.join(main_folder, item)
            if os.path.isdir(patient_folder) and item != "Dataset":
                logging.info(f"Processing folder: {patient_folder}")
                organize_files(patient_folder, images_dir, labels_dir, landmarks_dir)

    except Exception as e:
        logging.error(f"Error processing main folder {main_folder}: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <main_folder_path>")
        logging.error("Incorrect number of arguments provided")
        sys.exit(1)

    main_folder = sys.argv[1]
    if not os.path.isdir(main_folder):
        print(f"Error: {main_folder} is not a valid directory.")
        logging.error(f"Invalid directory provided: {main_folder}")
        sys.exit(1)

    logging.info(f"Starting processing of main folder: {main_folder}")
    try:
        process_main_folder(main_folder)
        logging.info("Processing completed successfully.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during processing: {str(e)}")
    print(f"Processing complete. Log file: {log_filename}")