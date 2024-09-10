import os
import argparse
import csv
import json
import numpy as np
import SimpleITK as sitk
import re

def preprocess_image_and_labels(input_path, output_path, label_path, label_format):
    print(f'Processing image: {input_path}')

    # Read the input image
    image = sitk.ReadImage(input_path)
    
    # Get the current image size and spacing
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    original_origin = image.GetOrigin()

    # Calculate the physical size of the image
    physical_size = [sz*spc for sz, spc in zip(original_size, original_spacing)]

    # Calculate the new spacing
    new_spacing = [phys_sz / 256 for phys_sz in physical_size]

    print(f"Original size: {original_size}")
    print(f"Original spacing: {original_spacing}")
    print(f"Original origin: {original_origin}")
    print(f"New spacing: {new_spacing}")

    print("[1] Resampling the image...")
    resampled_image = sitk.Resample(
        image,
        [256, 256, 256],
        sitk.Transform(),
        sitk.sitkLinear,
        original_origin,
        new_spacing,
        image.GetDirection(),
        0,
        image.GetPixelID()
    )

    print("[2] Clipping and normalizing...")
    numpy_image = sitk.GetArrayFromImage(resampled_image)
    numpy_image = np.clip(numpy_image, -1000, np.quantile(numpy_image, 0.999))
    numpy_image = (numpy_image - np.min(numpy_image)) / (np.max(numpy_image) - np.min(numpy_image))

    assert numpy_image.shape == (256, 256, 256), "The output shape should be (256, 256, 256)"

    print(f"[3] FINAL REPORT: Min value: {numpy_image.min()}, "
          f"Max value: {numpy_image.max()}, Shape: {numpy_image.shape}")

    # Convert back to SimpleITK image
    processed_image = sitk.GetImageFromArray(numpy_image)
    processed_image.SetSpacing(new_spacing)
    processed_image.SetOrigin(original_origin)
    processed_image.SetDirection(image.GetDirection())

    # Save the processed image
    sitk.WriteImage(processed_image, output_path)

    # Process labels
    if label_path:
        labels = load_labels(label_path)
        transformed_labels = transform_labels(labels, image, processed_image)
        
        # Save transformed labels
        if label_format == 'csv':
            label_output_path = output_path.replace('.nii.gz', '_labels.csv').replace('.nrrd', '_labels.csv')
            save_labels_csv(transformed_labels, label_output_path)
        else:  # label_format == 'mrk.json'
            label_output_path = output_path.replace('.nii.gz', '.mrk.json').replace('.nrrd', '.mrk.json')
            save_labels_mrk_json(transformed_labels, label_output_path)

    print("-------------------------------------------------------------------------------")

def load_labels(label_path):
    if label_path.endswith('.json'):
        with open(label_path, 'r') as f:
            data = json.load(f)
            return [(point['label'], point['position']) for point in data['markups'][0]['controlPoints']]
    elif label_path.endswith('.csv'):
        with open(label_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            return [(row[0], [float(row[1]), float(row[2]), float(row[3])]) for row in reader]
    else:
        raise ValueError("Unsupported label file format")

def transform_labels(labels, original_image, processed_image):
    original_size = original_image.GetSize()
    original_spacing = original_image.GetSpacing()
    original_origin = original_image.GetOrigin()
    
    new_size = processed_image.GetSize()
    new_spacing = processed_image.GetSpacing()
    new_origin = processed_image.GetOrigin()

    print(f"Original size: {original_size}, spacing: {original_spacing}, origin: {original_origin}")
    print(f"New size: {new_size}, spacing: {new_spacing}, origin: {new_origin}")

    transformed_labels = []
    for i, (name, position) in enumerate(labels, start=1):
        # Convert world coordinates to index coordinates
        original_index = [int((p - o) / s) for p, o, s in zip(position, original_origin, original_spacing)]
        
        # Scale index coordinates to new image size
        new_index = [int(idx * (nsz / osz)) for idx, nsz, osz in zip(original_index, new_size, original_size)]
        
        # Convert back to world coordinates
        new_position = [idx * s + o for idx, s, o in zip(new_index, new_spacing, new_origin)]

        print(f"Label {name}: Original position: {position}, Original index: {original_index}, "
              f"New index: {new_index}, New position: {new_position}")

        transformed_labels.append({
            "id": str(i),
            "label": name,
            "description": "",
            "associatedNodeID": "",
            "position": new_position,
            "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
            "selected": True,
            "locked": False,
            "visibility": True,
            "positionStatus": "defined"
        })
    
    return transformed_labels

def save_labels_mrk_json(labels, output_path):
    markup_json = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "coordinateUnits": "mm",
                "locked": False,
                "fixedNumberOfControlPoints": False,
                "labelFormat": "%N-%d",
                "lastUsedControlPointNumber": len(labels),
                "controlPoints": labels
            }
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(markup_json, f, indent=2)

def save_labels_csv(labels, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'x', 'y', 'z'])
        for name, position in labels:
            writer.writerow([name] + position)

def process_directory(input_dir, output_dir, label_dir, label_format):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a dictionary to store label file paths
    label_files = {}
    for label_file in os.listdir(label_dir):
        match = re.match(r'(BS-\d+)', label_file)
        if match:
            label_files[match.group(1)] = os.path.join(label_dir, label_file)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith((".nii", ".nii.gz", ".mha", ".mhd", ".nrrd")):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                output_dir_path = os.path.dirname(output_path)
                
                if not os.path.exists(output_dir_path):
                    os.makedirs(output_dir_path)
                
                # Match the label file using the BS-XXX identifier
                match = re.match(r'(BS-\d+)', file)
                if match:
                    label_identifier = match.group(1)
                    label_path = label_files.get(label_identifier)
                    
                    if label_path:
                        print(f"Processing {file} with label file: {os.path.basename(label_path)}")
                        try:
                            preprocess_image_and_labels(input_path, output_path, label_path, label_format)
                        except Exception as e:
                            print(f"Error processing {input_path} with label {label_path}:")
                            print(f"Error details: {str(e)}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"Label file not found for {file}. Processing image without labels.")
                        try:
                            preprocess_image_and_labels(input_path, output_path, None, label_format)
                        except Exception as e:
                            print(f"Error processing {input_path} without labels:")
                            print(f"Error details: {str(e)}")
                            import traceback
                            traceback.print_exc()
                else:
                    print(f"Could not extract identifier from {file}. Skipping.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess medical images.")
    DEFAULT_INPUT_DIR = "/home/juval.gutknecht/Projects/Data/A_Subset_012_a/imagesTs"
    DEFAULT_OUTPUT_DIR = "/home/juval.gutknecht/Projects/Data/A_Subset_012_a/imagesTs_preprocessed"
    DEFAULT_LABEL_DIR = "/home/juval.gutknecht/Projects/Data/A_Subset_012_a/landmarksTs_csv"

    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR,
                        help=f'Directory containing the original image data (default: {DEFAULT_INPUT_DIR})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory to store the processed image files (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--label_dir', type=str, default=DEFAULT_LABEL_DIR,
                        help=f'Directory containing the label files (default: {DEFAULT_LABEL_DIR})')
    parser.add_argument('--label_format', type=str, choices=['csv', 'mrk.json'], default='csv',
                        help='Format to save the labels (default: csv)')

    args = parser.parse_args()

    print(f"Processing images from: {args.input_dir}")
    print(f"Saving processed images to: {args.output_dir}")
    print(f"Looking for labels in: {args.label_dir}")
    print(f"Saving labels in {args.label_format} format")

    process_directory(args.input_dir, args.output_dir, args.label_dir, args.label_format)
