import os
import json


'''This program scans directories containing medical imaging files in the NRRD format
and generates a JSON file. This JSON file can be used as a configuration for data loaders
in machine learning pipelines, specifying the paths to the volume images and their corresponding
segmentation labels.'''

def generate_json(directory):
    volumes_dir = os.path.join(directory, "TrainVolumes")
    segmentations_dir = os.path.join(directory, "TrainSegmentations")

    # Initialize the structure of the output JSON
    output = {"training": []}

    # List all files in the Volumes and Segmentations directories
    volume_files = [f for f in os.listdir(volumes_dir) if f.endswith('.nrrd')]
    segmentation_files = [f for f in os.listdir(segmentations_dir) if f.endswith('.seg.nrrd')]

    # Create a dictionary to match volumes with their corresponding segmentations
    for volume_file in volume_files:
        base_name = volume_file.replace('.nrrd', '')
        segmentation_file = base_name + '.seg.nrrd'
        if segmentation_file in segmentation_files:
            # Construct the training entry
            training_entry = {
                "fold": 0,
                "image": [os.path.join(volumes_dir, volume_file)],
                "label": os.path.join(segmentations_dir, segmentation_file)
            }
            # Append to the output
            output["training"].append(training_entry)
    
    # Define the output JSON file path
    json_output_path = os.path.join(directory, "USB.json")
    
    # Write the JSON structure to a file
    with open(json_output_path, 'w') as json_file:
        json.dump(output, json_file, indent=4)

    print(f"JSON file successfully created at {json_output_path}")

# Example usage
generate_json('/home/juval.gutknecht/Projects/Data/USB')