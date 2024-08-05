import os
import sys
import nrrd
import numpy as np
import tqdm
import time
import logging
from datetime import datetime

# Set up logging
log_filename = f"label_changer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def create_label_mapping():
    current_labels = {
        1: "LVOT",
        2: "RCC",
        3: "LCC",
        4: "NCC",
        5: "BG"
    }
    
    desired_labels = {
        0: "BG",
        1: "LVOT",
        2: "RCC",
        3: "LCC",
        4: "NCC"
    }

    label_mapping = {
        1: "LVOT",
        2: "RCC",
        3: "LCC",
        4: "NCC",
        5: "BG"
    }
    reverse_mapping = {v: k for k, v in desired_labels.items()}

    return current_labels, desired_labels, label_mapping, reverse_mapping

def detailed_label_comparison(original_data, modified_data, label_mapping, reverse_mapping):
    mismatches = {}
    for old_label, segment_name in label_mapping.items():
        new_label = reverse_mapping[segment_name]
        original_indices = (original_data == old_label)
        modified_indices = (modified_data == new_label)
        
        if not np.array_equal(original_indices, modified_indices):
            original_count = np.sum(original_indices)
            modified_count = np.sum(modified_indices)
            mismatch_indices = np.where(original_indices != modified_indices)
            mismatch_count = len(mismatch_indices[0])
            
            mismatches[segment_name] = {
                'old_label': old_label,
                'new_label': new_label,
                'original_count': original_count,
                'modified_count': modified_count,
                'mismatch_count': mismatch_count,
                'mismatch_indices': mismatch_indices
            }
    
    if mismatches:
        logging.warning(f"Label mismatches found:")
        for segment_name, mismatch in mismatches.items():
            logging.warning(f"  {segment_name} (old: {mismatch['old_label']}, new: {mismatch['new_label']}):")
            logging.warning(f"    Original count: {mismatch['original_count']}, Modified count: {mismatch['modified_count']}")
            logging.warning(f"    Total mismatched voxels: {mismatch['mismatch_count']}")
            logging.warning(f"    Sample of mismatched voxels:")
            for i in range(min(5, mismatch['mismatch_count'])):
                x, y, z = mismatch['mismatch_indices'][0][i], mismatch['mismatch_indices'][1][i], mismatch['mismatch_indices'][2][i]
                logging.warning(f"      At ({x},{y},{z}): Original={original_data[x,y,z]}, Modified={modified_data[x,y,z]}")

def adjust_labels(input_file, label_mapping, reverse_mapping):
    try:
        start_time = time.time()
        data, header = nrrd.read(input_file)
        read_time = time.time() - start_time

        start_time = time.time()
        new_header = {}
        segment_order = ['BG', 'LVOT', 'RCC', 'LCC', 'NCC']
        
        for new_index, segment_name in enumerate(segment_order):
            for old_index in range(5):
                if header.get(f"Segment{old_index}_Name") == segment_name:
                    for subkey in [k for k in header if k.startswith(f"Segment{old_index}_")]:
                        new_key = f"Segment{new_index}_{subkey[9:]}"
                        new_header[new_key] = header[subkey]
                        if subkey.endswith("LabelValue"):
                            new_header[new_key] = str(new_index)
                    break

        for key, value in header.items():
            if not key.startswith("Segment"):
                new_header[key] = value
        header_time = time.time() - start_time

        start_time = time.time()
        adjusted_data = np.copy(data)
        copy_time = time.time() - start_time

        start_time = time.time()
        for old_label, segment_name in label_mapping.items():
            new_label = reverse_mapping[segment_name]
            adjusted_data[data == old_label] = new_label
        label_mapping_time = time.time() - start_time

        start_time = time.time()
        nrrd.write(input_file, adjusted_data, new_header)
        write_time = time.time() - start_time

        return read_time, header_time, copy_time, label_mapping_time, write_time

    except Exception as e:
        logging.error(f"Error adjusting labels in file {input_file}: {str(e)}")
        return None

def check_labels_changed(original_data, input_file, label_mapping, reverse_mapping):
    try:
        modified_data, _ = nrrd.read(input_file)
        mismatches = detailed_label_comparison(original_data, modified_data, label_mapping, reverse_mapping)
        
        if not mismatches:
            logging.info(f"Labels in {os.path.basename(input_file)} have been changed correctly.")
            return True
        else:
            logging.warning(f"Labels in {os.path.basename(input_file)} have NOT been changed correctly.")
            return False
    except Exception as e:
        logging.error(f"Error verifying labels in file {input_file}: {str(e)}")
        return False

def process_folder(folder):
    current_labels, desired_labels, label_mapping, reverse_mapping = create_label_mapping()
    
    for filename in tqdm.tqdm(os.listdir(folder)):
        if filename.endswith(".nrrd"):
            input_file = os.path.join(folder, filename)
            
            try:
                # Read the original data for verification
                original_data, _ = nrrd.read(input_file)
                
                result = adjust_labels(input_file, label_mapping, reverse_mapping)
                if result is not None:
                    read_time, header_time, copy_time, label_mapping_time, write_time = result
                    logging.info(f"Processed {filename}: Read: {read_time:.2f}s, Header: {header_time:.2f}s, Copy: {copy_time:.2f}s, Label Mapping: {label_mapping_time:.2f}s, Write: {write_time:.2f}s")
                
                # Verify the changes
                if not check_labels_changed(original_data, input_file, label_mapping, reverse_mapping):
                    logging.warning(f"Verification failed for {filename}. Manual inspection may be required.")
            except Exception as e:
                logging.error(f"Error processing file {input_file}: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python change_label.py /path/to/Dataset/labelsTr")
        sys.exit(1)

    input_folder = sys.argv[1]
    if not os.path.isdir(input_folder):
        print(f"Error: {input_folder} is not a valid directory.")
        logging.error(f"Invalid directory provided: {input_folder}")
        sys.exit(1)

    print(f"Warning: This script will modify the .nrrd files in {input_folder} in-place.")
    print("Make sure you have a backup of your data before proceeding.")
    confirmation = input("Do you want to continue? (y/n): ")

    if confirmation.lower() != 'y':
        print("Operation cancelled.")
        logging.info("Operation cancelled by user.")
        sys.exit(0)

    logging.info(f"Starting processing of folder: {input_folder}")
    try:
        process_folder(input_folder)
        logging.info("Processing completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during processing: {str(e)}")
        print(f"An error occurred. Please check the log file {log_filename} for details.")
    print(f"Processing complete. Log file: {log_filename}")
