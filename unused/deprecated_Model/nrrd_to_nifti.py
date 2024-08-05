import nrrd
import nibabel as nib
import numpy as np
import os

def convert_nrrd_to_nii_gz(input_file, output_file):
    # Read the NRRD file
    data, header = nrrd.read(input_file)
    
    # Create a NIfTI image
    nifti_img = nib.Nifti1Image(data, np.eye(4))
    
    # Save the NIfTI image as a .nii.gz file
    nib.save(nifti_img, output_file)
    print(f"Conversion complete: {output_file}")

if __name__ == "__main__":
    input_path = '/home/juval.gutknecht/Projects/Data/BS-004/11 Herz  0.6  I26f  3  60%.nrrd'
    output_path = '/home/juval.gutknecht/Projects/Data/BS-004/11 Herz  0.6  I26f  3  60%.nii.gz'
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert the file
    convert_nrrd_to_nii_gz(input_path, output_path)