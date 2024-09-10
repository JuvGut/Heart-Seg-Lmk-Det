import argparse
import SimpleITK as sitk
import numpy as np
import os
import pandas as pd

import sys
sys.path.append("..")
sys.path.append(".")

from detection3d.utils.image_tools import resample_spacing
from detection3d.utils.landmark_utils import is_world_coordinate_valid, is_voxel_coordinate_valid

def gen_single_landmark_mask(ref_image, landmark_df, spacing, pos_upper_bound, neg_lower_bound):
    """
    Generate landmark mask for a single image.
    """
    print(f"Generating single landmark mask with spacing: {spacing}, pos_upper_bound: {pos_upper_bound}, neg_lower_bound: {neg_lower_bound}")
    assert isinstance(ref_image, sitk.Image)

    ref_image = resample_spacing(ref_image, spacing, 1, 'NN')
    ref_image_npy = sitk.GetArrayFromImage(ref_image)
    ref_image_size = ref_image.GetSize()
    print(f"Resampled image size: {ref_image_size}")
    
    landmark_mask_npy = np.zeros_like(ref_image_npy)
    for landmark_name in landmark_df.keys():
        print(f"Processing landmark: {landmark_name}")
        landmark_label = landmark_df[landmark_name]['label']
        landmark_world = [landmark_df[landmark_name]['x'],
                          landmark_df[landmark_name]['y'],
                          landmark_df[landmark_name]['z']]
        landmark_voxel = ref_image.TransformPhysicalPointToIndex(landmark_world)
        print(f"  World coordinate: {landmark_world}")
        print(f"  Voxel coordinate: {landmark_voxel}")
        
        # ... (rest of the function remains the same)

    landmark_mask = sitk.GetImageFromArray(landmark_mask_npy)
    landmark_mask.CopyInformation(ref_image)
    print(f"Generated landmark mask with size: {landmark_mask.GetSize()}")
    return landmark_mask

def gen_landmark_mask_batch(image_folder, landmark_folder, target_landmark_label,
                            spacing, pos_upper_bound, neg_lower_bound, landmark_mask_save_folder):
    """
    Generate landmark mask for a batch of images
    """
    print(f"Generating landmark masks for batch:")
    print(f"  Image folder: {image_folder}")
    print(f"  Landmark folder: {landmark_folder}")
    print(f"  Save folder: {landmark_mask_save_folder}")
    
    # Get landmark file list
    landmark_files = [f for f in os.listdir(landmark_folder) if f.endswith('.csv')]
    print(f"Found {len(landmark_files)} landmark files")

    # Get image file list (look for .nrrd files)
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.nrrd')]
    print(f"Found {len(image_files)} image files")

    # Match landmark files with image files
    matched_files = []
    for landmark_file in landmark_files:
        landmark_prefix = landmark_file.split('_')[0]  # Get the BS-XXX part
        matching_images = [img for img in image_files if img.startswith(landmark_prefix)]
        if matching_images:
            matched_files.append((landmark_file, matching_images[0]))
        else:
            print(f"Warning: No matching image found for landmark file {landmark_file}")

    print(f"Matched {len(matched_files)} landmark-image pairs")

    if not os.path.isdir(landmark_mask_save_folder):
        os.makedirs(landmark_mask_save_folder)
        print(f"Created save folder: {landmark_mask_save_folder}")

    for landmark_file, image_file in matched_files:
        print(f"\nProcessing pair:")
        print(f"  Landmark: {landmark_file}")
        print(f"  Image: {image_file}")

        landmark_file_path = os.path.join(landmark_folder, landmark_file)
        landmark_df = pd.read_csv(landmark_file_path)
        print(f"Loaded landmark data with {len(landmark_df)} entries")
        
        target_landmark_df = {}
        for landmark_name in target_landmark_label.keys():
            landmark_name_clean = landmark_name.strip().lower()
            matching_landmarks = landmark_df[landmark_df['name'].str.strip().str.lower().str.contains(landmark_name_clean)]
            if not matching_landmarks.empty:
                x = matching_landmarks['x'].values[0]
                y = matching_landmarks['y'].values[0]
                z = matching_landmarks['z'].values[0]
                if is_world_coordinate_valid([x, y, z]):
                    target_landmark_df[landmark_name] = {
                        'label': target_landmark_label[landmark_name],
                        'x': float(x),
                        'y': float(y),
                        'z': float(z)
                    }
                    print(f"Added landmark {landmark_name} with coordinates ({x}, {y}, {z})")
                else:
                    print(f"Warning: Invalid world coordinate for landmark {landmark_name}: ({x}, {y}, {z})")
            else:
                print(f"Warning: Landmark {landmark_name} not found in the CSV file")

        image_file_path = os.path.join(image_folder, image_file)
        print(f"Reading image from: {image_file_path}")
        image = sitk.ReadImage(image_file_path)
        print(f"Image size: {image.GetSize()}, spacing: {image.GetSpacing()}")
        
        landmark_mask = gen_single_landmark_mask(
            image, target_landmark_df, spacing, pos_upper_bound, neg_lower_bound
        )

        save_path = os.path.join(landmark_mask_save_folder, f'{os.path.splitext(image_file)[0]}_landmark_mask.nii.gz')
        print(f"Saving landmark mask to: {save_path}")
        sitk.WriteImage(landmark_mask, save_path, True)

    print(f"Completed processing {len(matched_files)} files")

def generate_landmark_mask(image_folder, landmark_folder, landmark_label_file, spacing, bound, save_folder):
    """
    Generate landmark mask.
    """
    # Trim whitespace from input paths
    image_folder = image_folder.strip()
    landmark_folder = landmark_folder.strip()
    landmark_label_file = landmark_label_file.strip()
    save_folder = save_folder.strip()

    print(f"Generating landmark masks:")
    print(f"  Image folder: {image_folder}")
    print(f"  Landmark folder: {landmark_folder}")
    print(f"  Landmark label file: {landmark_label_file}")
    print(f"  Spacing: {spacing}")
    print(f"  Bounds: {bound}")
    print(f"  Save folder: {save_folder}")
    
    landmark_label_df = pd.read_csv(landmark_label_file)
    print(f"Loaded landmark label file with {len(landmark_label_df)} entries")
    
    target_landmark_label = {}
    for row in landmark_label_df.iterrows():
        target_landmark_label.update({row[1]['landmark_name']: row[1]['landmark_label']})
    print(f"Target landmarks: {list(target_landmark_label.keys())}")

    pos_upper_bound, neg_lower_bound = bound[0], bound[1]
    gen_landmark_mask_batch(image_folder, landmark_folder, target_landmark_label, spacing,
                            pos_upper_bound, neg_lower_bound, save_folder)

def main():
    long_description = 'Generate landmark mask for landmark detection.'

    default_input = '/home/juval.gutknecht/Projects/Data/Dataset012_USB_only_Heart_aligned/labelsTr'
    default_landmark = '/home/juval.gutknecht/Projects/Data/Dataset012_USB_only_Heart_aligned/landmarksTr_csv'
    default_output = '/home/juval.gutknecht/Projects/Data/Dataset012_USB_only_Heart_aligned/landmark/mask_label'
    default_label = '/home/juval.gutknecht/Projects/Data/Dataset012_USB_only_Heart_aligned/landmark/landmark_label_file.csv'
    default_spacing = [0.43, 0.3, 0.43] # [1.3, 0.85, 1.3]
    default_pos_upper_bound = 3
    default_neg_lower_bound = 6

    parser = argparse.ArgumentParser(description=long_description)
    parser.add_argument('-i', '--input', default=default_input,
                        help='input folder for intensity images.')
    parser.add_argument('-l', '--landmark', default=default_landmark,
                        help='landmark folder.')
    parser.add_argument('-o', '--output', default=default_output,
                        help='output folder for the landmark mask')
    parser.add_argument('-n', '--label', default=default_label,
                        help='the label file containing the selected landmark names.')
    parser.add_argument('-s', '--spacing', default=default_spacing,
                        help='the spacing of the landmark mask.')
    parser.add_argument('-b', '--bound', default=[default_pos_upper_bound, default_neg_lower_bound],
                        help='the pos. upper bound and the neg. lower bound of the landmark mask.')

    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    generate_landmark_mask(args.input.strip(), args.landmark.strip(), args.label.strip(), args.spacing, args.bound, args.output.strip())

if __name__ == '__main__':
    print("Starting landmark mask generation script")
    main()
    print("Landmark mask generation complete")