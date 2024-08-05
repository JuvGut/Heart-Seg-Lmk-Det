import os

def rename_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".nrrd"):
            new_filename = filename.replace(".seg.nrrd", ".nrrd") # 1st: how it is, 2nd: how you want it to be
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

# Replace "/path/to/folder" with the actual path to your folder
folder_path = "/home/juval.gutknecht/Projects/nnUNet_raw/Dataset011_USB_Heart_small/labelsTs"
rename_files(folder_path)
