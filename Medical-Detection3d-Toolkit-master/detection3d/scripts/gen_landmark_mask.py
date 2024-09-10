import argparse
import SimpleITK as sitk
import numpy as np
import os
import pandas as pd
import glob

import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from detection3d.utils.image_tools import resample_spacing
from detection3d.utils.landmark_utils import is_world_coordinate_valid, is_voxel_coordinate_valid
from detection3d.vis.gen_images import gen_plane_images, load_coordinates_from_csv
from detection3d.vis.gen_html_report import gen_html_report

def normalize_image(image, lower_percentile=0.5, upper_percentile=99.5):
    """
    Normalize the image by clipping to percentiles and scaling to [0, 1].
    """
    array = sitk.GetArrayFromImage(image)
    lower = np.percentile(array, lower_percentile)
    upper = np.percentile(array, upper_percentile)
    array = np.clip(array, lower, upper)
    array = (array - lower) / (upper - lower)
    normalized_image = sitk.GetImageFromArray(array)
    normalized_image.CopyInformation(image)
    return normalized_image

def gen_single_landmark_mask(ref_image, landmark_df, spacing, pos_upper_bound, neg_lower_bound, debug=False):
    """
    Generate landmark mask for a single image.
    """
    if debug:
        print(f"Generating single landmark mask with spacing: {spacing}, pos_upper_bound: {pos_upper_bound}, neg_lower_bound: {neg_lower_bound}")
    assert isinstance(ref_image, sitk.Image)

    # Normalize the image
    # ref_image = normalize_image(ref_image)

    ref_image = resample_spacing(ref_image, spacing, 1, 'NN')
    ref_image_npy = sitk.GetArrayFromImage(ref_image)
    ref_image_size = ref_image.GetSize()
    if debug:
        print(f"Resampled image size: {ref_image_size}")

    # Initialize the mask with background value
    landmark_mask_npy = np.zeros_like(ref_image_npy) #, dtype=np.float32)

    for landmark_name in landmark_df.keys():
        if debug:
            print(f"Processing landmark: {landmark_name}")
        landmark_label = landmark_df[landmark_name]['label']
        landmark_world = [landmark_df[landmark_name]['x'],
                          landmark_df[landmark_name]['y'],
                          landmark_df[landmark_name]['z']]
        landmark_voxel = ref_image.TransformPhysicalPointToIndex(landmark_world)
        if debug:
            print(f"  World coordinate: {landmark_world}")
            print(f"  Voxel coordinate: {landmark_voxel}")

        # search_range = max(pos_upper_bound, neg_lower_bound)
        # for x in range(landmark_voxel[0] - search_range,
        #                landmark_voxel[0] + search_range):
        #     for y in range(landmark_voxel[1] - search_range,
        #                    landmark_voxel[1] + search_range):
        #         for z in range(landmark_voxel[2] - search_range,
        #                        landmark_voxel[2] + search_range):
        #             if is_voxel_coordinate_valid([x, y, z], ref_image_size):
        #                 distance = np.linalg.norm(np.array([x, y, z], dtype=np.float32) - landmark_voxel)
        #                 if distance < pos_upper_bound:
        #                     # Set positive sample value (landmark)
        #                     landmark_mask_npy[z, y, x] = float(landmark_label)
        #                 elif neg_lower_bound > distance > pos_upper_bound:
        #                     # Set negative sample value
        #                     landmark_mask_npy[z, y, x] = 0.5
        for x in range(landmark_voxel[0] - neg_lower_bound,
                   landmark_voxel[0] + neg_lower_bound):
            for y in range(landmark_voxel[1] - neg_lower_bound,
                     landmark_voxel[1] + neg_lower_bound):
                for z in range(landmark_voxel[2] - neg_lower_bound,
                       landmark_voxel[2] + neg_lower_bound):
                    if is_voxel_coordinate_valid([x, y, z], ref_image_size):
                        distance = np.linalg.norm(np.array([x, y, z], dtype=np.float32) - landmark_voxel)
                        if distance < pos_upper_bound:
                            landmark_mask_npy[z, y, x] = float(landmark_label)
                        elif distance < neg_lower_bound and abs(landmark_mask_npy[z, y, x]) < 1e-6:
                            landmark_mask_npy[z, y, x] = -1.0

    landmark_mask = sitk.GetImageFromArray(landmark_mask_npy)
    landmark_mask.CopyInformation(ref_image)
    if debug:
        print(f"Generated landmark mask with size: {landmark_mask.GetSize()}")
    return landmark_mask

def gen_landmark_mask_batch(image_folder, landmark_folder, target_landmark_label,
                            spacing, pos_upper_bound, neg_lower_bound, landmark_mask_save_folder, debug=False):
    """
    Generate landmark mask for a batch of images
    """
    if debug:
        print(f"Generating landmark masks for batch:")
        print(f"  Image folder: {image_folder}")
        print(f"  Landmark folder: {landmark_folder}")
        print(f"  Save folder: {landmark_mask_save_folder}")

    landmark_files = glob.glob(os.path.join(landmark_folder, "BS-*.csv"))
    image_files = glob.glob(os.path.join(image_folder, "BS-*.nrrd"))

    if debug:
        print(f"Found {len(landmark_files)} landmark files")
        print(f"Found {len(image_files)} image files")

    matched_files = []
    for landmark_file in landmark_files:
        case_id = os.path.basename(landmark_file).split('_')[0]  # Get the BS-XXX part
        matching_images = [img for img in image_files if os.path.basename(img).startswith(case_id)]
        if matching_images:
            matched_files.append((landmark_file, matching_images[0]))
        elif debug:
            print(f"Warning: No matching image found for landmark file {landmark_file}")

    if debug:
        print(f"Matched {len(matched_files)} landmark-image pairs")

    if not os.path.isdir(landmark_mask_save_folder):
        os.makedirs(landmark_mask_save_folder)
        if debug:
            print(f"Created save folder: {landmark_mask_save_folder}")

    for landmark_file, image_file in matched_files:
        case_id = os.path.basename(landmark_file).split('_')[0]  # Get the BS-XXX part
        print(f"Processing: {case_id}")

        if debug:
            print(f"\nProcessing pair:")
            print(f"  Landmark: {landmark_file}")
            print(f"  Image: {image_file}")

        landmarks = load_coordinates_from_csv(landmark_file)
        if debug:
            print(f"Loaded landmark data with {len(landmarks)} entries")
            print("Landmarks found in CSV:")
            for lm_name, lm_coords in landmarks.items():
                print(f"  {lm_name}: {lm_coords}")

        target_landmark_df = {}
        if debug:
            print("\nMatching landmarks:")
        for landmark_name in target_landmark_label.keys():
            landmark_name_clean = landmark_name.strip()
            if debug:
                print(f"Looking for landmark: {landmark_name_clean}")
            if landmark_name_clean in landmarks:
                x, y, z = landmarks[landmark_name_clean]
                if is_world_coordinate_valid([x, y, z]):
                    target_landmark_df[landmark_name_clean] = {
                        'label': target_landmark_label[landmark_name],
                        'x': float(x),
                        'y': float(y),
                        'z': float(z)
                    }
                    if debug:
                        print(f"  Added landmark {landmark_name_clean} with coordinates ({x}, {y}, {z})")
                elif debug:
                    print(f"  Warning: Invalid world coordinate for landmark {landmark_name_clean}: ({x}, {y}, {z})")
            elif debug:
                print(f"  Warning: Landmark {landmark_name_clean} not found in the CSV file")

        if debug:
            print(f"\nReading image from: {image_file}")
        image = sitk.ReadImage(image_file)
        if debug:
            print(f"Image size: {image.GetSize()}, spacing: {image.GetSpacing()}")

        landmark_mask = gen_single_landmark_mask(
            image, target_landmark_df, spacing, pos_upper_bound, neg_lower_bound, debug
        )

        save_path = os.path.join(landmark_mask_save_folder, f'{os.path.splitext(os.path.basename(image_file))[0]}_landmark_mask.nii.gz')
        if debug:
            print(f"Saving landmark mask to: {save_path}")
        sitk.WriteImage(landmark_mask, save_path, True)

    print(f"Completed processing {len(matched_files)} files")

def generate_landmark_mask(image_folder, landmark_folder, landmark_label_file, spacing, bound, save_folder, debug=False):
    """
    Generate landmark mask.
    """
    if debug:
        print(f"Generating landmark masks:")
        print(f"  Image folder: {image_folder}")
        print(f"  Landmark folder: {landmark_folder}")
        print(f"  Landmark label file: {landmark_label_file}")
        print(f"  Spacing: {spacing}")
        print(f"  Bounds: {bound}")
        print(f"  Save folder: {save_folder}")

    landmark_label_df = pd.read_csv(landmark_label_file)
    if debug:
        print(f"Loaded landmark label file with {len(landmark_label_df)} entries")

    target_landmark_label = {}
    for row in landmark_label_df.iterrows():
        target_landmark_label.update({row[1]['landmark_name'].strip(): row[1]['landmark_label']})
    if debug:
        print(f"Target landmarks: {list(target_landmark_label.keys())}")

    pos_upper_bound, neg_lower_bound = bound[0], bound[1]
    gen_landmark_mask_batch(image_folder, landmark_folder, target_landmark_label, spacing,
                            pos_upper_bound, neg_lower_bound, save_folder, debug)

def main():
    long_description = 'Generate landmark mask for landmark detection.'

    default_image_folder = '/home/juval.gutknecht/Projects/Data/A_Subset_012_a/imagesTr' # imagesTr
    default_landmark_folder = '/home/juval.gutknecht/Projects/Data/A_Subset_012_a/landmarksTr_csv' # landmarksTr_csv
    default_output_folder = '/home/juval.gutknecht/Projects/Data/A_Subset_012_a/landmark/batch_1' # where to save the landmark masks
    default_label_file = '/home/juval.gutknecht/Projects/Data/A_Subset_012_a/landmark/landmark_label_file_1.csv' # label_file.csv, containing the landmark names and labels
    default_spacing = [0.43, 0.3, 0.43] # [0.43, 0.3, 0.43]
    default_pos_upper_bound = 3 # default_pos_upper_bound = 3
    default_neg_lower_bound = 6 # default_neg_lower_bound = 6
    default_generate_pictures = False

    parser = argparse.ArgumentParser(description=long_description)
    parser.add_argument('-i', '--image_folder', default=default_image_folder,
                        help='input folder for intensity images.')
    parser.add_argument('-l', '--landmark_folder', default=default_landmark_folder,
                        help='landmark folder.')
    parser.add_argument('-o', '--output_folder', default=default_output_folder,
                        help='output folder for the landmark mask')
    parser.add_argument('-n', '--label_file', default=default_label_file,
                        help='the label file containing the selected landmark names.')
    parser.add_argument('-s', '--spacing', nargs=3, type=float, default=default_spacing,
                        help='the spacing of the landmark mask.')
    parser.add_argument('-b', '--bound', nargs=2, type=float, default=[default_pos_upper_bound, default_neg_lower_bound],
                        help='the pos. upper bound and the neg. lower bound of the landmark mask in voxels.')
    parser.add_argument('--generate_pictures', type=bool, default=default_generate_pictures,
                        help='Generate snapshot images of landmarks.')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode for verbose output.')

    args = parser.parse_args()

    if args.debug:
        print("Arguments:")
        for arg in vars(args):
            print(f"  {arg}: {getattr(args, arg)}")

    print("Starting landmark mask generation")
    generate_landmark_mask(args.image_folder, args.landmark_folder, args.label_file, 
                           args.spacing, args.bound, args.output_folder, args.debug)

    if args.generate_pictures:
        print('Start generating planes for the landmarks.')
        label_landmarks = {}
        for landmark_file in glob.glob(os.path.join(args.landmark_folder, "BS-*.csv")):
            file_name = os.path.basename(landmark_file).split('.')[0]
            landmarks = load_coordinates_from_csv(landmark_file)
            label_landmarks.update({file_name: landmarks})

        gen_plane_images(args.image_folder, label_landmarks, 'label',
                         None, args.spacing, args.output_folder)

if __name__ == '__main__':
    main()
    print("Landmark mask generation complete")