import SimpleITK as sitk
import numpy as np
import os

def fix_label_headers(path_img, path_lab):
    # Read NRRD images
    image = sitk.ReadImage(path_img, imageIO="NrrdImageIO")
    label = sitk.ReadImage(path_lab, imageIO="NrrdImageIO")

    # Get numpy arrays
    arr_image = sitk.GetArrayFromImage(image)
    arr_label = sitk.GetArrayFromImage(label)

    # Check and correct orientation if necessary
    if arr_label.shape != arr_image.shape:
        print(f"Original shapes - Image: {arr_image.shape}, Label: {arr_label.shape}")
        for i in range(3):
            for j in range(i+1, 3):
                if arr_label.shape[i] == arr_image.shape[j] and arr_label.shape[j] == arr_image.shape[i]:
                    arr_label = np.swapaxes(arr_label, i, j)
                    print(f"Swapped axes {i} and {j}")
                    break
            if arr_label.shape == arr_image.shape:
                break
        print(f"Final shapes - Image: {arr_image.shape}, Label: {arr_label.shape}")

    if arr_label.shape != arr_image.shape:
        raise ValueError("Unable to align label shape with image shape")

    # Round label values to nearest integer
    arr_label = np.round(arr_label).astype(np.int32)

    # Create new label image with corrected orientation and metadata
    new_label = sitk.GetImageFromArray(arr_label)
    new_label.CopyInformation(image)

    print("Image metadata:", image.GetSize(), image.GetOrigin(), image.GetSpacing(), image.GetDirection())
    print("Label metadata:", new_label.GetSize(), new_label.GetOrigin(), new_label.GetSpacing(), new_label.GetDirection())

    return image, new_label

if __name__ == "__main__":
    path_img = '/home/juval.gutknecht/Projects/Data/heart-segmentation-results/BS-093/5 Fl_Th-Abd Art  3.0  I31f  3 cropped.nrrd'
    path_lab = '/home/juval.gutknecht/Projects/Data/heart-segmentation-results/BS-093/Segmentation.seg.nrrd'
    
    image, aligned_label = fix_label_headers(path_img, path_lab)
    
    # Optionally, save the aligned images
    # sitk.WriteImage(image, 'aligned_image.nrrd')
    # sitk.WriteImage(aligned_label, 'aligned_label.nrrd')




# # fixed alignement between labels and mri
# import SimpleITK as sitk
# import numpy as np
# import os
# def fix_label_headers(path_img, path_lab):
#     image = sitk.ReadImage(os.path.join(path_img), imageIO="NiftiImageIO")
#     label = sitk.ReadImage(os.path.join(path_lab), imageIO="NiftiImageIO")
#     arr1 = sitk.GetArrayFromImage(label)
#     arr2 = sitk.GetImageFromArray(image)
#     if arr1.shape[0] != arr2.shape[0] and arr1.shape[1] != arr2.shape[1]:
#         arr1 = np.swapaxes(arr1, 0, 1)
#     elif arr1.shape[0] != arr2.shape[0] and arr1.shape[2] != arr2.shape[2]:
#         arr1 = np.swapaxes(arr1, 0, 2)
#     elif arr1.shape[2] != arr2.shape[2] and arr1.shape[1] != arr2.shape[1]:
#         arr1 = np.swapaxes(arr1, 1, 2)
#     else:
#         pass
#     arr = np.round(arr1, 0)
#     label = sitk.GetImageFromArray(arr)
#     label.SetDirection(image.GetDirection())
#     label.SetSpacing(image.GetSpacing())
#     label.SetOrigin(image.GetOrigin())
#     image.SetDirection(image.GetDirection())
#     image.SetSpacing(image.GetSpacing())
#     image.SetOrigin(image.GetOrigin())
#     print(image.GetSize(), image.GetOrigin(), image.GetSpacing(), image.GetDirection())
#     print(label.GetSize(), label.GetOrigin(), label.GetSpacing(), image.GetDirection())
#     return image, label

# if __name__ == "__main__":
#     path_img = '/home/helenecorbaz/storage_server/users/helene.corbaz/Stroke/perf/0002052001/30702232/mri/dwi.nii.gz'
#     path_lab = '/home/helenecorbaz/storage_server/users/helene.corbaz/Stroke/perf/0002072551/30858925/mri/Seg.nii.gz'
#     # sitk.WriteImage(label, os.path.join(path_img, 'DWI_' + i + '_0001.nii.gz'))
#     # sitk.WriteImage(image, os.path.join(path_img, 'DWI_' + i + '_0000.nii.gz'))