import os
import json
import sys
from datetime import datetime

def create_json_structure(root_dir):
    # Define the structure
    structure = {
        "name": "HeartScans",
        "description": "Heart segmentation",
        "tensorImageSize": "3D",
        "reference": "University Hospital Basel",
        "licence": "??????",
        "relase": f"1.0 {datetime.now().strftime('%d/%m/%Y')}",
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0,
            "LVOT": 1,
            "RCC": 2,
            "LCC": 3,
            "NCC": 4
        },
        "numTraining": 0,
        "file_ending": ".nrrd"
    }

    # Count training data
    training_images_dir = os.path.join(root_dir, "imagesTr")
    
    if os.path.exists(training_images_dir):
        structure["numTraining"] = len([f for f in os.listdir(training_images_dir) if f.endswith(".nrrd")])

    return structure

def save_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

def main(root_dir):
    if not os.path.exists(root_dir):
        print(f"Error: The directory {root_dir} does not exist.")
        return
    
    output_file = os.path.join(root_dir, "dataset.json")

    json_structure = create_json_structure(root_dir)
    save_json(json_structure, output_file)
    print(f"JSON file '{output_file}' has been created successfully.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_dataset_json.py /path/to/Dataset")
        sys.exit(1)
    
    main(sys.argv[1])