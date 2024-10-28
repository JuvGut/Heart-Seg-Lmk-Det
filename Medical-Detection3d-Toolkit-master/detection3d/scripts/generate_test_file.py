import os
import csv
import argparse

def generate_csv_from_nrrd(input_folder, output_file):
    # Ensure the input folder path is absolute
    input_folder = os.path.abspath(input_folder)
    
    # Create a list to store the data
    data = []
    
    # Walk through the input folder
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.nrrd'):
                # Get the full path of the file
                file_path = os.path.join(root, file)
                
                # Get the filename without extension as the image_name
                image_name = os.path.splitext(file)[0]
                
                # Add the data to our list
                data.append([image_name, file_path])
    
    # Check if the output file already exists
    file_exists = os.path.isfile(output_file)
    
    # Open the file in append mode if it exists, otherwise in write mode
    mode = 'a' if file_exists else 'w'
    
    with open(output_file, mode, newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write the header only if the file is newly created
        if not file_exists:
            csv_writer.writerow(['image_name', 'image_path'])
        
        # Write the data
        csv_writer.writerows(data)

    print(f"Data has been {'appended to' if file_exists else 'written to'} '{output_file}' successfully.")

if __name__ == "__main__":
    # Set default values
    default_input_folder = os.path.join(os.path.expanduser("~"), "/home/juval.gutknecht/Projects/Data/Dataset012_aligned/imagesTs")
    default_output_file = os.path.join(os.path.expanduser("~"), "/home/juval.gutknecht/Projects/Data/Dataset012_aligned/test.csv")

    parser = argparse.ArgumentParser(description="Generate or append to CSV from NRRD files in a folder")
    parser.add_argument("-i", "--input_folder", default=default_input_folder, 
                        help=f"Path to the input folder containing NRRD files (default: {default_input_folder})")
    parser.add_argument("-o", "--output_file", default=default_output_file, 
                        help=f"Path to the output CSV file (default: {default_output_file})")
    
    args = parser.parse_args()

    generate_csv_from_nrrd(args.input_folder, args.output_file)