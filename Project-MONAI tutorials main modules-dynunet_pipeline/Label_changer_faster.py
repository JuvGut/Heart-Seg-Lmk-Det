import os
import nrrd
import numpy as np
import tqdm as tqdm
import time

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

def adjust_labels(input_file, output_file, label_mapping, reverse_mapping):
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
    # Optimized label mapping using NumPy vectorized operations
    for old_label, segment_name in label_mapping.items():
        new_label = segment_order.index(segment_name)
        adjusted_data = np.where(data == old_label, new_label, adjusted_data)
    label_mapping_time = time.time() - start_time

    start_time = time.time()
    nrrd.write(output_file, adjusted_data, new_header)
    write_time = time.time() - start_time

    return read_time, header_time, copy_time, label_mapping_time, write_time

def check_labels_changed(original_file, modified_file, label_mapping, reverse_mapping):
    original_data, _ = nrrd.read(original_file)
    modified_data, _ = nrrd.read(modified_file)
    
    labels_changed = True
    for old_label, segment_name in label_mapping.items():
        new_label = reverse_mapping[segment_name]
        original_indices = (original_data == old_label)
        modified_indices = (modified_data == new_label)
        if not np.array_equal(original_indices, modified_indices):
            labels_changed = False
            print(f"Label {segment_name} (old label: {old_label}, new label: {new_label}) does not match in {os.path.basename(modified_file)}")
    
    if labels_changed:
        print(f"Labels in {os.path.basename(modified_file)} have been changed correctly.")
    else:
        print(f"Labels in {os.path.basename(modified_file)} have NOT been changed correctly.")

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    current_labels, desired_labels, label_mapping, reverse_mapping = create_label_mapping()

    for filename in tqdm.tqdm(os.listdir(input_folder)):
        if filename.endswith(".nrrd"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)
            
            read_time, header_time, copy_time, label_mapping_time, write_time = adjust_labels(input_file, output_file, label_mapping, reverse_mapping)
            
            # print(f"\nProcessing file: {filename}")
            # print(f"Read time: {read_time:.4f} seconds")
            # print(f"Header time: {header_time:.4f} seconds")
            # print(f"Copy time: {copy_time:.4f} seconds")
            # print(f"Label mapping time: {label_mapping_time:.4f} seconds")
            # print(f"Write time: {write_time:.4f} seconds")
        check_labels_changed(input_file, output_file, label_mapping, reverse_mapping)

if __name__ == "__main__":
    input_folder = '/home/juval.gutknecht/Projects/Data/USB/USB_Heart/old_labelsTr'
    output_folder = '/home/juval.gutknecht/Projects/Data/USB/USB_Heart/labelsTr'

    process_folder(input_folder, output_folder)