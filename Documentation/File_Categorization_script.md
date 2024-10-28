# File Categorization Script Documentation

## Overview

This script helps organize medical image cutouts by automatically grouping similar images into separate folders. While **not essential for the main project functionality**, it serves as a useful preprocessing tool to organize your image dataset.

The script analyzes file names to identify different types of medical images (e.g., heart scans, thorax images, angiograms) and systematically organizes them into categorized folders for easier access and management.
Features

- Categorizes files into four categories: Heart (H), Thorax (T), Angio (A), and Other (O).
- Allows selection of specific categories to process.
- Provides option to specify a custom output folder.
- Creates new folders only for categories that contain files.
- Provides a summary of processed files and created folders.

## Requirements

- Python 3.x
- No additional libraries required (uses only built-in modules)

## Usage

### Basic Syntax

```
python categorize_files.py <source_folder> [options]
```

### Command-line Arguments

- `source_folder` (required): Path to the folder containing files to categorize.
- `-o, --output`: Path to the output folder (optional). If not specified, uses the same directory as the source folder.
- `-c, --categories`: Categories to process (optional). Can be any combination of H, T, A, O. If not specified, processes all categories.

### Examples

1. Process all categories in the default output location:
   ```
   python categorize_files.py /path/to/source/folder
   ```

2. Process only Heart and Thorax files, with a custom output folder:
   ```
   python categorize_files.py /path/to/source/folder -o /path/to/output/folder -c H T
   ```

3. Process Angio and Other files in the default output location:
   ```
   python categorize_files.py /path/to/source/folder -c A O
   ```

## File Categorization Rules

- **Heart (H)**: Files containing "Herz" in the filename.
- **Thorax (T)**: Files containing any of these terms: "Thorax", "Thoracica", "Thx", "Abdomen", "Abd".
- **Angio (A)**: Files containing "Angio" but not any of the Thorax terms.
- **Other (O)**: All other files not matching the above criteria.

## Output

- Creates folders named `{source_folder_name}_{category}` for each processed category.
- Copies files into their respective category folders.
- Prints a summary of files processed for each category.
- Lists the names of folders created.

## Functions

### `categorize_files(source_folder)`

Scans the source folder and categorizes files based on their names.

### `create_category_folders(source_folder, output_folder, categorized_files, selected_categories)`

Creates new folders for selected categories and copies files into them.

### `main()`

Handles command-line arguments and orchestrates the file categorization process.

## Error Handling

- Checks if the source folder exists before processing.
- Creates the output folder if it doesn't exist.

## Limitations

- Does not process files in subdirectories of the source folder.
- Overwrites files in destination folders if they have the same name as source files.

## Future Enhancements

- Add option to process subdirectories recursively.
- Implement file conflict resolution strategies.
- Add logging for more detailed operation tracking.