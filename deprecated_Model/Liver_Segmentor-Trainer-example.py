# imports 
from glob import glob
import shutil
import os
import dicom2nifti


# create new folders with 64 files each
in_path = "/home/juval.gutknecht/Projects/Heart-Valve-Segmentation/Liver"
out_path = "/home/juval.gutknecht/Projects/Heart-Valve-Segmentation/Liver"

for patient in glob(in_path + "/*"):
    patient_name = os.path.basename(patient)
    number_folders = int(len(glob(patient + "/*"))/64)
    
    for i in range(number_folders):
        output_path_name = os.path.join(out_path, patient_name + '_' + str(i))
        os.mkdir(output_path_name)
        for i, file in enumerate(glob(patient + "/*")):
            if i == 64 + 1:
                break
            shutil.move(file, output_path_name)

# Convert the dicom files to nifties

in_path_images = "/home/juval.gutknecht/Projects/Heart-Valve-Segmentation/Liver/dicom_groups/images/*"
in_path_labels = "/home/juval.gutknecht/Projects/Heart-Valve-Segmentation/Liver/dicom_groups/labels/*"
out_path_images = "/home/juval.gutknecht/Projects/Heart-Valve-Segmentation/Liver/nifti_files/images"
out_path_labels = "/home/juval.gutknecht/Projects/Heart-Valve-Segmentation/Liver/nifti_files/labels"

list_images = glob(in_path_images)
list_labels = glob(in_path_labels)

for patient in list_images:
    patient_name = os.path.basename(os.path.normpath(patient))
    dicom2nifti.dicom_series_to_nifti(patient, os.path.join(out_path_images, patient_name + ".nii.gz"))




