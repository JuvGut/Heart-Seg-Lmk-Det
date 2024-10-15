import os
import SimpleITK as sitk

def resample_image(image, new_spacing):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    new_size = [
        int(round(osz*ospc/nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkLinear)
    
    return resample.Execute(image)

def process_folder(input_folder, output_folder, new_spacing):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.nii', '.nii.gz', '.nrrd', '.dcm')):  # Add or modify extensions as needed
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            print(f"Processing: {filename}")
            
            image = sitk.ReadImage(input_path)
            resampled_image = resample_image(image, new_spacing)
            
            sitk.WriteImage(resampled_image, output_path)
            
            print(f"Processed and saved: {filename}")
            print(f"Original spacing: {image.GetSpacing()}")
            print(f"New spacing: {resampled_image.GetSpacing()}")
            print("----------------------------")


# Example usage
input_folder = "/home/juval.gutknecht/Projects/Data/Dataset012_USB_Heart_big/imagesTr_fs"
output_folder = "/home/juval.gutknecht/Projects/Data/Dataset012_USB_Heart_big/imagesTr"
new_spacing = (1,1,1)  # New spacing in mm

process_folder(input_folder, output_folder, new_spacing)