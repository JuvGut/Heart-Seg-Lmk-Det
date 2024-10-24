"""
Script for preprocessing the LIDC-IDRI dataset.

This script is of no use for the landmark detection algorithm, because it only resamples, resizes, the images
and doesn't change anything in the annotations. Use the pre-align_images.py script.
"""
import argparse
import os
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom


def preprocess_image(input_path, output_path):
    # Load the image
    print('Process image: {}'.format(input_path))
    # img = nib.load(input_path)
    img = sitk.ReadImage(input_path)

    # Get the current voxel sizes
    voxel_sizes = img.GetSpacing()

    # Calculate the target voxel size (1mm x 1mm x 1mm)
    target_voxel_size = (1.0, 1.0, 1.0)

    # Calculate the resampling factor
    zoom_factors = [current / target for target, current in zip(target_voxel_size, voxel_sizes)]

    # Resample the image
    print("[1] Resample the image ...")
    img_array = sitk.GetArrayFromImage(img)
    resampled_data = zoom(img_array, zoom_factors, order=3, mode='nearest')

    print("[2] Center crop the image ...")
    crop_size = (128, 128, 128)#(256, 256, 256)
    depth, height, width = resampled_data.shape

    # Ensure we can crop to the desired size
    if any(curr < target for curr, target in zip(resampled_data.shape, crop_size)):
        raise ValueError(f"Image too small after resampling. Got {resampled_data.shape}, need {crop_size}")
    
    d_start = max(0, (depth - crop_size[0]) // 2)
    h_start = max(0, (height - crop_size[1]) // 2)
    w_start = max(0, (width - crop_size[2]) // 2)
    cropped_arr = resampled_data[d_start:d_start + crop_size[0], 
                                 h_start:h_start + crop_size[1], 
                                 w_start:w_start + crop_size[2]]
    print(f"After cropping shape: {cropped_arr.shape}")

    print("[3] Clip all values below -1000 ...")
    # cropped_arr[cropped_arr < -1000] = -1000
    cropped_arr = np.clip(cropped_arr, -1000, None)
    print(f"Value range after lower clip: [{np.min(cropped_arr)}, {np.max(cropped_arr)}]")

    print("[4] Clip the upper quantile (0.999) to remove outliers ...")
    #out_clipped = np.clip(cropped_arr, -1000, np.quantile(cropped_arr, 0.999))
    upper_quantile = np.quantile(cropped_arr, 0.999)
    out_clipped = np.clip(cropped_arr, -1000, upper_quantile)
    print(f"Value range after upper clip: [{np.min(out_clipped)}, {np.max(out_clipped)}]")

    print("[5] Normalize the image ...")
    # out_normalized = (out_clipped - np.min(out_clipped)) / (np.max(out_clipped) - np.min(out_clipped))
    min_val = np.min(out_clipped)
    max_val = np.max(out_clipped)
    if max_val == min_val:
        print("Warning: Constant image values, setting to zeros")
        out_normalized = np.zeros_like(out_clipped)
    else:
        out_normalized = (out_clipped - min_val) / (max_val - min_val)

    print("Value range after normalization: [{np.min(out_normalized)}, {np.max(out_normalized)}]")
    
    # Verify the final shape
    # assert out_normalized.shape == (256, 256, 256), "The output shape should be (320,320,320)"
    if out_normalized.shape != crop_size:
        raise ValueError(f"Final shape is not correct. Expected {crop_size}, got {out_normalized.shape}")
    

    print("[6] FINAL REPORT: Min value: {}, Max value: {}, Shape: {}".format(out_normalized.min(),
                                                                             out_normalized.max(),
                                                                             out_normalized.shape))
    print("-------------------------------------------------------------------------------")
    # Save the resampled image

    output_img = sitk.GetImageFromArray(out_normalized.astype(np.float32))
    output_img.SetDirection(img.GetDirection())
    output_img.SetSpacing(target_voxel_size)
    output_img.SetOrigin(img.GetOrigin())
    
    if output_path.endswith('.nrrd'):
        sitk.WriteImage(output_img, output_path, useCompression=True)
    else:
       # For other formats like .nii.gz 
        sitk.WriteImage(output_img, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing the original dicom data')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to store the processed nifti files')
    parser.add_argument('--output_format', type=str, default='nrrd',
                        help='Output format (.nrrd or .nii.gz)')
    args = parser.parse_args()

    # Ensure output format starts with a dot
    if not args.output_format.startswith('.'):
        args.output_format = '.' + args.output_format

    # Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Process all files in input directory
    processed_count = 0
    error_count = 0

    # Preprocess files
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.nrrd'):
                try:
                    rel_path = os.path.relpath(root, args.input_dir)
                    output_dir = os.path.join(args.output_dir, rel_path)
                    os.makedirs(output_dir, exist_ok=True)
                    
                    filename_base = os.path.splitext(file)[0]

                    input_path = os.path.join(root, file)
                    output_path = os.path.join(output_dir, f'{filename_base}{args.output_format}')

                    preprocess_image(input_path, output_path)
                    processed_count += 1
    
                except Exception as e:
                    print(f"Error occurred for file: {file}")
                    print(f"Error message: {str(e)}")
                    error_count += 1
    print(f"Processed {processed_count} files with {error_count} errors")