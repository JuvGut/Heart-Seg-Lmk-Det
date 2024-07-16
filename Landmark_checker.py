import os
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(filename='folder_checker.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def check_folders(root_path):
    folders_with_file = []
    folders_without_file = []
    target_files = ["MarkupsFiducial.mrk.json", "Landmarks.mrk.json"]
    
    try:
        for folder_name in os.listdir(root_path):
            folder_path = os.path.join(root_path, folder_name)
            
            if os.path.isdir(folder_path):
                file_found = False
                for target_file in target_files:
                    file_path = os.path.join(folder_path, target_file)
                    if os.path.exists(file_path):
                        file_found = True
                        break
                
                if file_found:
                    folders_with_file.append(folder_name)
                else:
                    folders_without_file.append(folder_name)
    except Exception as e:
        logging.error(f"Error occurred while checking folders: {str(e)}")
        raise
    
    return folders_with_file, folders_without_file

def write_results(folders_with_file, folders_without_file):
    results = {
        "folders_with_file": folders_with_file,
        "folders_without_file": folders_without_file
    }
    
    try:
        with open("folder_check_results.json", "w") as f:
            json.dump(results, f, indent=4)
        logging.info("Results successfully written to 'folder_check_results.json'")
    except Exception as e:
        logging.error(f"Error occurred while writing results: {str(e)}")
        raise

def main():
    try:
        root_path = input("Enter the path of the root folder to check: ")
        
        if not os.path.isdir(root_path):
            logging.error(f"Invalid directory path provided: {root_path}")
            print("The provided path is not a valid directory.")
            return
        
        logging.info(f"Starting folder check in: {root_path}")
        folders_with_file, folders_without_file = check_folders(root_path)
        
        write_results(folders_with_file, folders_without_file)
        
        print(f"Folders with either 'MarkupsFiducial.mrk.json' or 'Landmarks.mrk.json': {len(folders_with_file)}")
        print(f"Folders without either file: {len(folders_without_file)}")
        print("Results have been written to 'folder_check_results.json'")
        logging.info("Folder check completed successfully")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.error(f"An error occurred in main execution: {str(e)}")

if __name__ == "__main__":
    main()