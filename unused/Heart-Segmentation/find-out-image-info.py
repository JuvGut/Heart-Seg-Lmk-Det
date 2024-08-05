import os
import SimpleITK as sitk

def get_image_info(image_path):
    # Read the image
    image = sitk.ReadImage(image_path)
    
    # Get the numpy array for intensity values
    array = sitk.GetArrayFromImage(image)
    
    # Compute min and max intensities
    min_intensity = array.min()
    max_intensity = array.max()
    
    # Get spatial size (in voxels)
    size = image.GetSize()
    
    # Get spacing (physical units per voxel)
    spacing = image.GetSpacing()
    
    # Compute physical size
    physical_size = [s*sp for s, sp in zip(size, spacing)]
    
    return min_intensity, max_intensity, size, physical_size

def print_info(image_path):
    min_int, max_int, size, physical_size = get_image_info(image_path)
    print(f"Volume file information:")
    print(f"  Intensity range: {min_int} to {max_int}")
    print(f"  Size (voxels): {size}")
    print(f"  Physical size (mm): {[round(ps, 2) for ps in physical_size]}")


# Read the .nrrd file
image_path = "/home/juval.gutknecht/Projects/Data/BS-005/11 Herz  0.6  I26f  3  BestDiast 67 %.nrrd"
print_info(image_path)
