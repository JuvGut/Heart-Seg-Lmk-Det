import os
import json
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
        "modality": {
            "0": "CT"
        },
        "labels": {
            "0": "LVOT",
            "1": "RCC",
            "2": "LCC",
            "3": "NCC",
            "4": "background"
        },
        "numTraining": 0,
        "numTest": 0,
        "training": [],
        "test": []
    }

    # Process training data
    training_images_dir = os.path.join(root_dir, "imagesTr")
    training_labels_dir = os.path.join(root_dir, "labelsTr")
    
    for image_file in os.listdir(training_images_dir):
        if image_file.endswith(".nrrd"):
            image_path = os.path.join("./imagesTr", image_file)
            label_file = image_file.replace(".nrrd", ".seg.nrrd")
            label_path = os.path.join("./labelsTr", label_file)
        
            if os.path.exists(os.path.join(training_labels_dir, label_file)):
                structure["training"].append({
                    "image": image_path,
                    "label": label_path
                })
                structure["numTraining"] += 1

    # Process test data
    test_images_dir = os.path.join(root_dir, "imagesTs")
    
    for image_file in os.listdir(test_images_dir):
        image_path = os.path.join("./imagesTs", image_file)
        structure["test"].append(image_path)
        structure["numTest"] += 1

    return structure

def save_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    root_dir = input("Enter the root directory path: ")
    
    if not os.path.exists(root_dir):
        print(f"Error: The directory {root_dir} does not exist.")
        return
    
    output_file = os.path.join(root_dir, "dataset.json")

    json_structure = create_json_structure(root_dir)
    save_json(json_structure, output_file)
    print(f"JSON file '{output_file}' has been created successfully.")

if __name__ == "__main__":
    main()