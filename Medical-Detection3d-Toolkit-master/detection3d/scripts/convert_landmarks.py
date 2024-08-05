import json
import csv
import sys
import os

def convert_landmark_file(input_file, output_file):
    # Read the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Extract landmarks
    landmarks = []
    for markup in data['markups']:
        if markup['type'] == 'Fiducial':
            for point in markup['controlPoints']:
                name = point['label']
                x, y, z = point['position']
                landmarks.append({
                    'name': name,
                    'x': x,
                    'y': y,
                    'z': z
                })

    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'x', 'y', 'z'])
        writer.writeheader()
        for landmark in landmarks:
            writer.writerow(landmark)

def process_folder(input_folder):
    # Create output folder outside the input folder
    parent_folder = os.path.dirname(input_folder)
    input_folder_name = os.path.basename(input_folder)
    output_folder = os.path.join(parent_folder, f"{input_folder_name}_csv")
    os.makedirs(output_folder, exist_ok=True)
    # Process each .mrk.json file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mrk.json"):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(os.path.splitext(filename)[0])[0] + ".csv"
            output_path = os.path.join(output_folder, output_filename)
            
            convert_landmark_file(input_path, output_path)
            print(f"Converted {filename} to {output_filename}")

    print(f"All conversions complete. Output files are in the folder: {output_folder}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py input_folder")
        sys.exit(1)

    input_folder = sys.argv[1]
    process_folder(input_folder)