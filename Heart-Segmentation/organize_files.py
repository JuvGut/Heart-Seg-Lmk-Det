import os
import shutil
import sys

def organize_files(patient_folder, volumes_dir, segmentations_dir):
    files = os.listdir(patient_folder)

    volume_file = next((f for f in files if f.endswith('.nrrd') and not f.endswith('.seg.nrrd')), None)
    if volume_file:
        source_path = os.path.join(patient_folder, volume_file)
        dest_path = os.path.join(volumes_dir, f"{os.path.basename(patient_folder)}.nrrd")
        shutil.copy2(source_path, dest_path)
        print(f"Copied volume file: {volume_file} -> {dest_path}")
    else:
        print("No volume file found.")

    seg_file = next((f for f in files if f.endswith('.seg.nrrd')), None)
    if seg_file:
        source_path = os.path.join(patient_folder, seg_file)
        # dest_path = os.path.join(segmentations_dir, f"{os.path.basename(patient_folder)}.seg.nrrd")
        dest_path = os.path.join(segmentations_dir, f"{os.path.basename(patient_folder)}.nrrd")
        shutil.copy2(source_path, dest_path)
        print(f"Copied segmentation file: {seg_file} -> {dest_path}")
    else:
        print("No segmentation file found.")

def process_main_folder(main_folder):
    volumes_dir = os.path.join(main_folder, "Volumes")
    segmentations_dir = os.path.join(main_folder, "Segmentations")
    os.makedirs(volumes_dir, exist_ok=True)
    os.makedirs(segmentations_dir, exist_ok=True)

    for item in os.listdir(main_folder):
        patient_folder = os.path.join(main_folder, item)
        if os.path.isdir(patient_folder) and item not in ["Volumes", "Segmentations"]:
            print(f"Processing folder: {patient_folder}")
            organize_files(patient_folder, volumes_dir, segmentations_dir)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <main_folder_path>")
        sys.exit(1)

    main_folder = sys.argv[1]
    process_main_folder(main_folder)