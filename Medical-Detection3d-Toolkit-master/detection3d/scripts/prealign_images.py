import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import csv
import json
from tqdm import tqdm


def set_new_origin(image):
    """
    Set the origin of a SimpleITK image to (0, 0, 0).
    
    :param image: SimpleITK Image object
    :return: SimpleITK Image with origin set to (0, 0, 0)
    """
    image.SetOrigin((0, 0, 0))
    return image


def read_landmarks(landmark_path):
    """
    Read landmarks from a 3D Slicer JSON file.
    
    :param landmark_path: Path to the landmark JSON file
    :return: Dictionary mapping landmark labels to their positions
    """
    with open(landmark_path, 'r') as lmk:
        data = json.load(lmk)
        return {cp['label']: cp['position'] for cp in data['markups'][0]['controlPoints']}


def write_landmarks(landmarks, output_path):
    """
    Write landmarks to a 3D Slicer compatible JSON file.
    
    :param landmarks: Dictionary mapping landmark labels to their positions
    :param output_path: Path where to save the landmark file
    """
    control_points = [{"id": str(i+1), "label": name, "description": "", "associatedNodeID": "",
                       "position": position, "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                       "selected": True, "locked": False, "visibility": True, "positionStatus": "defined"}
                      for i, (name, position) in enumerate(landmarks.items())]
    
    output_data = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "coordinateUnits": "mm",
                "locked": False,
                "fixedNumberOfControlPoints": False,
                "labelFormat": "%N-%d",
                "lastUsedControlPointNumber": len(control_points),
                "controlPoints": control_points
            }
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)    


def process_dataset(input_folder, output_folder, dataset_type):
    """
    Process a specific dataset (training or testing) by aligning images, segmentations,
    and landmarks to origin (0, 0, 0).
    
    :param input_folder: Path to the input folder containing images, labels, and landmarks
    :param output_folder: Path where to save the aligned data
    :param dataset_type: Either 'Tr' for training or 'Ts' for testing
    """
    # Define input and output folders
    images_in = os.path.join(input_folder, f'images{dataset_type}')
    segmentations_in = os.path.join(input_folder, f'labels{dataset_type}')
    landmarks_in = os.path.join(input_folder, f'landmarks{dataset_type}')
    
    images_out = os.path.join(output_folder, f'images{dataset_type}')
    segmentations_out = os.path.join(output_folder, f'labels{dataset_type}')
    landmarks_out = os.path.join(output_folder, f'landmarks{dataset_type}')

    # Check if the input folder exists
    if not os.path.exists(images_in):
        print(f"{dataset_type} dataset not found. Skipping...")
        return

    # Create output folders
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(segmentations_out, exist_ok=True)
    os.makedirs(landmarks_out, exist_ok=True)

    # Get list of image filenames
    image_files = [f for f in os.listdir(images_in) if f.endswith('_0000.nrrd')]
    for image_filename in tqdm(image_files, desc=f"Processing {dataset_type} images", unit="image"):
        base_filename = image_filename.replace('_0000.nrrd', '')
        
        # Process image
        image_path = os.path.join(images_in, image_filename)
        image = sitk.ReadImage(image_path)
        old_origin = np.array(image.GetOrigin())
        aligned_image = set_new_origin(image)
        sitk.WriteImage(aligned_image, os.path.join(images_out, image_filename), useCompression=True)

        # Process segmentation if it exists
        seg_filename = f"{base_filename}.nrrd"
        seg_path = os.path.join(segmentations_in, seg_filename)
        if os.path.exists(seg_path):
            segmentation = sitk.ReadImage(seg_path, sitk.sitkUInt8)
            aligned_seg = set_new_origin(segmentation)
            sitk.WriteImage(aligned_seg, os.path.join(segmentations_out, seg_filename), useCompression=True)

        # Process landmarks if they exist
        landmark_filename = f"{base_filename}.mrk.json"
        landmark_path = os.path.join(landmarks_in, landmark_filename)
        if os.path.exists(landmark_path):
            landmarks = read_landmarks(landmark_path)
            # aligned_landmarks = {}
            # for name, position in landmarks.items():
            #     old_position = np.array(position)
            #     aligned_position = np.array(position) - old_origin
            #     aligned_landmarks[name] = aligned_position.tolist()
            aligned_landmarks = {
                name: (np.array(position) - old_origin).tolist()
                for name, position in landmarks.items()
            }
            write_landmarks(aligned_landmarks, os.path.join(landmarks_out, landmark_filename))


def align_dataset(input_folder, output_folder):
    """
    Align a dataset by setting the origin of all images, segmentations, and landmarks to (0, 0, 0).
    Currently configured to process only testing dataset.
    
    :param input_folder: Path to the input folder containing the dataset
    :param output_folder: Path where to save the aligned dataset
    """
    # Process training and testing datasets
    # process_dataset(input_folder, output_folder, 'Tr')
    process_dataset(input_folder, output_folder, 'Ts')


# Example usage
if __name__ == "__main__":
    input_folder = "/home/juval.gutknecht/Projects/Data/A_Subset_012"
    output_folder = "/home/juval.gutknecht/Projects/Data/A_Subset_012_aligned"
    align_dataset(input_folder, output_folder)