import os
import sys
import shutil
import random
import logging
from datetime import datetime

# Set up logging
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)  # Create the log folder if it doesn't exist

log_filename = f"dataset_split_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(log_folder, log_filename)

logging.basicConfig(filename=log_filepath, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_folders(dataset_path):
    for folder in ['imagesTs', 'labelsTs', 'landmarksTs']:
        folder_path = os.path.join(dataset_path, folder)
        logging.info(f"Creating folder: {folder_path}")
        try:
            os.makedirs(folder_path, exist_ok=True)
        except Exception as e:
            logging.error(f"Error creating folder {folder_path}: {str(e)}")
            raise

def get_file_pairs(dataset_path):
    logging.info("Getting file pairs...")
    try:
        images = set(os.listdir(os.path.join(dataset_path, 'imagesTr')))
        labels = set(os.listdir(os.path.join(dataset_path, 'labelsTr')))
        landmarks = set(os.listdir(os.path.join(dataset_path, 'landmarksTr')))
    except Exception as e:
        logging.error(f"Error reading directory contents: {str(e)}")
        raise

    pairs = []
    for image in images:
        # Remove the '_0000.nrrd' suffix to get the base name
        base_name = image[:-10] if image.endswith('_0000.nrrd') else None
        if base_name is None:
            logging.warning(f"Skipping image with unexpected format: {image}")
            continue

        # Construct the corresponding label and landmark filenames
        label = f"{base_name}.nrrd"
        landmark = f"{base_name}.mrk.json"
        
        if label in labels and landmark in landmarks:
            pairs.append((image, label, landmark))
        else:
            logging.warning(f"Incomplete set for base name: {base_name}")

    logging.info(f"Found {len(pairs)} complete file pairs")
    return pairs

def move_files(dataset_path, file_pair):
    image, label, landmark = file_pair
    
    for file_name, folder_tr, folder_ts in [
        (image, 'imagesTr', 'imagesTs'),
        (label, 'labelsTr', 'labelsTs'),
        (landmark, 'landmarksTr', 'landmarksTs')
    ]:
        src = os.path.join(dataset_path, folder_tr, file_name)
        dst = os.path.join(dataset_path, folder_ts, file_name)
        logging.info(f"Moving {src} to {dst}")
        try:
            shutil.move(src, dst)
        except Exception as e:
            logging.error(f"Error moving file {src} to {dst}: {str(e)}")
            raise

def split_dataset(dataset_path, test_ratio=0.2):
    try:
        create_test_folders(dataset_path)
        file_pairs = get_file_pairs(dataset_path)
        
        num_test = int(len(file_pairs) * test_ratio)
        test_pairs = random.sample(file_pairs, num_test)
        
        logging.info(f"Moving {num_test} samples to test set")
        for pair in test_pairs:
            move_files(dataset_path, pair)
    except Exception as e:
        logging.error(f"Error in split_dataset: {str(e)}")
        raise

def main(dataset_path):
    if not os.path.exists(dataset_path):
        logging.error(f"The directory {dataset_path} does not exist.")
        print(f"Error: The directory {dataset_path} does not exist.")
        return

    try:
        split_dataset(dataset_path)
        logging.info("Dataset split completed successfully.")
        print("Dataset split completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during dataset split: {str(e)}")
        print(f"An error occurred. Check the log file {log_filename} for details.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python split_dataset.py /path/to/Dataset")
        sys.exit(1)
    
    main(sys.argv[1])