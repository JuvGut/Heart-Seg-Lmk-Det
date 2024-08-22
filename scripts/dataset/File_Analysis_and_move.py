import os
import shutil
import argparse

def categorize_files(source_folder):
    herz_files = []
    thorax_files = []
    angio_files = []
    other_files = []
    thorax_terms = ["Thorax", "Thoracica", "Thx", "Abdomen", "Abd"]

    for filename in os.listdir(source_folder):
        if "Herz" in filename:
            herz_files.append(filename)
        elif any(term in filename for term in thorax_terms):
            thorax_files.append(filename)
        elif "Angio" in filename and not any(term in filename for term in thorax_terms):
            angio_files.append(filename)
        else:
            other_files.append(filename)

    return {
        "H": herz_files,
        "T": thorax_files,
        "A": angio_files,
        "O": other_files
    }

def create_category_folders(source_folder, output_folder, categorized_files, selected_categories):
    folders_created = []
    for category, files in categorized_files.items():
        if category in selected_categories and files:
            new_folder = f"{os.path.basename(source_folder)}_{category}"
            new_folder_path = os.path.join(output_folder, new_folder)
            
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            
            for file in files:
                source_file = os.path.join(source_folder, file)
                destination_file = os.path.join(new_folder_path, file)
                shutil.copy2(source_file, destination_file)
            
            folders_created.append(new_folder)

    return folders_created

def main():
    parser = argparse.ArgumentParser(description="Categorize and copy files based on their names.")
    parser.add_argument("source_folder", help="Path to the source folder containing the files to categorize")
    parser.add_argument("-o", "--output", help="Path to the output folder (default: same as source folder)")
    parser.add_argument("-c", "--categories", choices=["H", "T", "A", "O"], nargs="+", 
                        default=["H", "T", "A", "O"], help="Categories to process (H: Heart, T: Thorax, A: Angio, O: Other)")

    args = parser.parse_args()

    source_folder = args.source_folder
    output_folder = args.output if args.output else os.path.dirname(source_folder)
    selected_categories = args.categories

    if not os.path.exists(source_folder):
        print("The specified source folder does not exist.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    categorized_files = categorize_files(source_folder)
    created_folders = create_category_folders(source_folder, output_folder, categorized_files, selected_categories)

    print("Files have been categorized and copied to new folders:")
    for category in selected_categories:
        files = categorized_files[category]
        if files:
            print(f"{category}: {len(files)} files")
        else:
            print(f"{category}: No files found")

    print("\nFolders created:")
    for folder in created_folders:
        print(folder)

if __name__ == "__main__":
    main()