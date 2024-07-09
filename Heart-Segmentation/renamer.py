import os

def rename_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".nrrd"):
            new_filename = filename.replace(".nrrd", ".seg.nrrd")
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

# Replace "/path/to/folder" with the actual path to your folder
folder_path = "/home/juval.gutknecht/Projects/Data/USB/TrainSegmentations"
rename_files(folder_path)