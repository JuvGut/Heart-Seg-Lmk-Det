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

    # Create a mapping from current to desired labels
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
    # Read the NRRD file
    start_time = time.time()
    data, header = nrrd.read(input_file)
    read_time = time.time() - start_time

    # Create a new header with reordered segments
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

    # Create a copy of the data to modify
    start_time = time.time()
    adjusted_data = np.copy(data)
    copy_time = time.time() - start_time

    # Apply the label mapping to the actual data -> this is a very slow process
    start_time = time.time()
    for old_label, segment_name in label_mapping.items(): 
        new_label = segment_order.index(segment_name)
        adjusted_data[data == old_label] = new_label
    label_mapping_time = time.time() - start_time

    # Save the adjusted labels to a new NRRD file
    start_time = time.time()
    nrrd.write(output_file, adjusted_data, new_header)
    write_time = time.time() - start_time

    return read_time, header_time, copy_time, label_mapping_time, write_time

def untimed_adjust_labels(input_file, output_file, label_mapping, reverse_mapping):
    # Read the NRRD file
    data, header = nrrd.read(input_file)
    
    # print("Processing file:", input_file)
    # print("Original header:")
    for key in sorted(header.keys()):
        if key.startswith("Segment") and not key.endswith("Tags"):
            # print(f"{key}: {header[key]}")
            pass
    
    # Create a new header with reordered segments
    new_header = {}
    segment_order = ['BG', 'LVOT', 'RCC', 'LCC', 'NCC']
    
    # # print("\nReordering segments:")
    for new_index, segment_name in enumerate(segment_order):
        # print(f"Processing {segment_name} to Segment{new_index}")
        for old_index in range(5):
            if header.get(f"Segment{old_index}_Name") == segment_name:
                for subkey in [k for k in header if k.startswith(f"Segment{old_index}_")]:
                    new_key = f"Segment{new_index}_{subkey[9:]}"
                    new_header[new_key] = header[subkey]
                    if subkey.endswith("LabelValue"):
                        new_header[new_key] = str(new_index)
                    # print(f"  {subkey} -> {new_key}: {new_header[new_key]}")
                break

    # Copy non-segment related headers
    for key, value in header.items():
        if not key.startswith("Segment"):
            new_header[key] = value

    # print("\nUpdated header:")
    for key in sorted(new_header.keys()):
        if key.startswith("Segment") and not key.endswith("Tags"):
            # print(f"{key}: {new_header[key]}")
            pass

    # Create a copy of the data to modify
    adjusted_data = np.copy(data)
    
    # print("\nApplying label mapping:")
    # Apply the label mapping to the actual data
    for old_label, segment_name in label_mapping.items():
        new_label = segment_order.index(segment_name)
        adjusted_data[data == old_label] = new_label
        # print(f"  {old_label} ({segment_name}) -> {new_label}")

    # Save the adjusted labels to a new NRRD file
    nrrd.write(output_file, adjusted_data, new_header)
    # print(f"\nProcessed {input_file} and saved to {output_file}")

    # Verify the changes
    # print("\nVerifying changes:")
    verified_data, verified_header = nrrd.read(output_file)
    for key in sorted(verified_header.keys()):
        if key.startswith("Segment") and not key.endswith("Tags"):
            # print(f"{key}: {verified_header[key]}")
            pass

    unique_labels = np.unique(verified_data)
    # print(f"Unique labels in the new data: {unique_labels}")

def check_labels_changed(original_file, modified_file, current_labels, desired_labels, label_mapping, reverse_mapping):
    # Read the original and modified NRRD files
    original_data, _ = nrrd.read(original_file)
    modified_data, _ = nrrd.read(modified_file)
    
    # Check if the labels have been changed correctly
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
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the label mappings
    current_labels, desired_labels, label_mapping, reverse_mapping = create_label_mapping()

    # Print out the mappings for verification
    print("Current Labels:", current_labels)
    print("Desired Labels:", desired_labels)
    # print("Label Mapping:", label_mapping)
    # print("Reverse Mapping:", reverse_mapping)

    # Process each NRRD file in the input folder
    for filename in tqdm.tqdm(os.listdir(input_folder)):
        if filename.endswith(".nrrd"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)
            
            # print(f"\nProcessing file: {filename}")
            read_time, header_time, copy_time, label_mapping_time, write_time = adjust_labels(input_file, output_file, label_mapping, reverse_mapping)

            print(f"\nProcessing file: {filename}")
            print(f"Read time: {read_time:.2f} seconds")
            print(f"Header time: {header_time:.2f} seconds")
            print(f"Copy time: {copy_time:.2f} seconds")
            print(f"Label mapping time: {label_mapping_time:.2f} seconds")
            print(f"Write time: {write_time:.2f} seconds")


if __name__ == "__main__":
    input_folder = '/home/juval.gutknecht/Projects/Data/USB/USB_Heart/labelsTs'
    output_folder = '/home/juval.gutknecht/Projects/Data/USB/USB_Heart/labelsTs_new'

    process_folder(input_folder, output_folder)