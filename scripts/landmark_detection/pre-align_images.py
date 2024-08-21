import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import csv
import json
from tqdm import tqdm

def set_new_origin(image):
    image.SetOrigin((0, 0, 0))
    return image

def read_landmarks(landmark_path):
    with open(landmark_path, 'r') as lmk:
        data = json.load(lmk)
        return {cp['label']: cp['position'] for cp in data['markups'][0]['controlPoints']}

def write_landmarks(landmarks, output_path):
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

def align_image_set(input_images, output_folder):
    """
    Align a set of images and save them to the output folder.
    
    :param input_images: List of paths to input image files
    :param output_folder: Path to the output folder
    """
    os.makedirs(output_folder, exist_ok=True)
    images = [sitk.ReadImage(img_path) for img_path in input_images]
    
    for i, img in enumerate(images):
        print(f"Original Image {i}: Size = {img.GetSize()}, Spacing = {img.GetSpacing()}, Origin = {img.GetOrigin()}")
    
    aligned_images = pre_align_images(images)
    
    for i, (aligned_img, input_path) in enumerate(zip(aligned_images, input_images)):
        output_filename = f"aligned_{os.path.basename(input_path)}"
        output_path = os.path.join(output_folder, output_filename)
        sitk.WriteImage(aligned_img, output_path)
        print(f"Saved aligned image to: {output_path}")
        print(f"Aligned Image {i}: Size = {aligned_img.GetSize()}, Spacing = {aligned_img.GetSpacing()}, Origin = {aligned_img.GetOrigin()}")

def process_dataset(input_folder, output_folder, dataset_type):
    """
    Process a specific dataset (training or testing).
    
    :param input_folder: Path to the input folder
    :param output_folder: Path to the output folder
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
        
        # Load and align image
        image_path = os.path.join(images_in, image_filename)
        image = sitk.ReadImage(image_path)
        old_origin = np.array(image.GetOrigin()) # store the old origin for landmark alignment (not (0,0,0))
        # print(f"old origin: {old_origin}") 
        aligned_image = set_new_origin(image) # set the new origin to (0,0,0)
        # print(f"new origin: {aligned_image.GetOrigin()}")
        sitk.WriteImage(aligned_image, os.path.join(images_out, image_filename), useCompression=True)

        # Load and align segmentation if it exists
        seg_filename = f"{base_filename}.nrrd"
        seg_path = os.path.join(segmentations_in, seg_filename)
        if os.path.exists(seg_path):
            segmentation = sitk.ReadImage(seg_path, sitk.sitkUInt8)
            aligned_seg = set_new_origin(segmentation)
            sitk.WriteImage(aligned_seg, os.path.join(segmentations_out, seg_filename), useCompression=True)

        # Load and align landmarks if they exist
        landmark_filename = f"{base_filename}.mrk.json"
        landmark_path = os.path.join(landmarks_in, landmark_filename)
        if os.path.exists(landmark_path):
            landmarks = read_landmarks(landmark_path)
            aligned_landmarks = {}
            for name, position in landmarks.items():
                old_position = np.array(position)
                aligned_position = np.array(position) - old_origin
                aligned_landmarks[name] = aligned_position.tolist()
                # print(f"Landmark: {name}")
                # print(f"Old position: {np.array2string(old_position, precision=2, separator=', ')}")
                # print(f"Aligned position: {np.array2string(aligned_position, precision=2, separator=', ')}")
                # print()
            write_landmarks(aligned_landmarks, os.path.join(landmarks_out, landmark_filename))

        # print(f"Processed {base_filename} and set to origin (0, 0, 0)")

def align_dataset(input_folder, output_folder):
    """
    Align a dataset by setting the origin of all images to (0, 0, 0).
    
    :param input_folder: Path to the input folder
    :param output_folder: Path to the output folder
    """
    # Process training and testing datasets
    process_dataset(input_folder, output_folder, 'Tr')
    process_dataset(input_folder, output_folder, 'Ts')


# Example usage
if __name__ == "__main__":
    input_folder = "/home/juval.gutknecht/Projects/Data/A_Subset_012"
    output_folder = "/home/juval.gutknecht/Projects/Data/A_Subset_012_a"
    align_dataset(input_folder, output_folder)



### bin (as in trash)

# def analyze_image_histogram(image, num_bins=256, percentile_low=5, percentile_high=95):
#     """
#     Analyze the histogram of a 3D image and suggest threshold values.
    
#     :param image: SimpleITK Image or path to the 3D image file
#     :param num_bins: Number of bins for the histogram
#     :param percentile_low: Lower percentile for threshold suggestion
#     :param percentile_high: Upper percentile for threshold suggestion
#     :return: Suggested lower and upper threshold values
#     """
#     if isinstance(image, str):
#         image = sitk.ReadImage(image)
    
#     array = sitk.GetArrayFromImage(image)
    
#     # Flatten the array and remove zero values (often background)
#     flat_array = array[array > 0]
    
#     # Calculate histogram
#     hist, bin_edges = np.histogram(flat_array, bins=num_bins)
    
#     # Calculate cumulative distribution
#     cumulative_dist = np.cumsum(hist) / np.sum(hist)
    
#     # Find threshold values based on percentiles
#     lower_threshold = np.interp(percentile_low/100, cumulative_dist, bin_edges[1:])
#     upper_threshold = np.interp(percentile_high/100, cumulative_dist, bin_edges[1:])
    
#     # Plot histogram and thresholds
#     plt.figure(figsize=(10, 6))
#     plt.hist(flat_array, bins=num_bins, density=True, alpha=0.7)
#     plt.axvline(lower_threshold, color='r', linestyle='dashed', linewidth=2)
#     plt.axvline(upper_threshold, color='r', linestyle='dashed', linewidth=2)
#     plt.title('Image Histogram with Suggested Thresholds')
#     plt.xlabel('Intensity')
#     plt.ylabel('Frequency')
#     plt.show()
#     plt.savefig('histogram.png')
    
#     return lower_threshold, upper_threshold

# def weighted_voxel_center(image, threshold_min, threshold_max):
#     """
#     Get the weighted voxel center.
#     :param image:
#     :return:
#     """
#     assert isinstance(image, sitk.Image)

#     image_npy = sitk.GetArrayFromImage(image)
#     image_npy[image_npy < threshold_min] = 0
#     image_npy[image_npy > threshold_max] = 0
#     weight_sum = np.sum(image_npy)
#     if weight_sum <= 0:
#         return None

#     image_npy_x = np.zeros_like(image_npy)
#     for i in range(image_npy.shape[0]):
#         image_npy_x[i , :, :] = i

#     image_npy_y = np.zeros_like(image_npy)
#     for j in range(image_npy.shape[1]):
#         image_npy_y[:, j, :] = j

#     image_npy_z = np.zeros_like(image_npy)
#     for k in range(image_npy.shape[2]):
#         image_npy_z[:, :, k] = k

#     weighted_center_x = np.sum(np.multiply(image_npy, image_npy_x)) / weight_sum
#     weighted_center_y = np.sum(np.multiply(image_npy, image_npy_y)) / weight_sum
#     weighted_center_z = np.sum(np.multiply(image_npy, image_npy_z)) / weight_sum
#     weighted_center = [weighted_center_z, weighted_center_y, weighted_center_x]

#     return weighted_center

# def get_physical_bounds(image):
#     """
#     Get the physical bounds of the image.
    
#     :param image: SimpleITK Image
#     :return: (min_point, max_point) tuple of physical coordinates
#     """
#     size = image.GetSize()
#     origin = image.GetOrigin()
#     spacing = image.GetSpacing()
#     direction = image.GetDirection()
    
#     # Calculate the corner points in index space
#     corners = [
#         (0, 0, 0),
#         (size[0]-1, 0, 0),
#         (0, size[1]-1, 0),
#         (0, 0, size[2]-1),
#         (size[0]-1, size[1]-1, 0),
#         (size[0]-1, 0, size[2]-1),
#         (0, size[1]-1, size[2]-1),
#         (size[0]-1, size[1]-1, size[2]-1)
#     ]
    
#     # Transform corners to physical space
#     physical_corners = [image.TransformIndexToPhysicalPoint(corner) for corner in corners]
    
#     # Find min and max coordinates
#     min_point = np.min(physical_corners, axis=0)
#     max_point = np.max(physical_corners, axis=0)
    
#     return min_point, max_point

# def pre_align_images(images):
#     """
#     Pre-align a list of 3D images by setting their origin to (0, 0, 0).
    
#     :param images: List of SimpleITK Image objects
#     :return: List of pre-aligned SimpleITK Image objects
#     """
#     aligned_images = []
    
#     for img in images:
#         # Create a new image with the same data
#         aligned_img = sitk.Image(img)
        
#         # Set the origin to (0, 0, 0)
#         aligned_img.SetOrigin((0, 0, 0))
        
#         aligned_images.append(aligned_img)
    
#     return aligned_images
