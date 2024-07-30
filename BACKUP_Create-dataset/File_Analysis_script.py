import os

def analyze_files(folder_path):
    herz_files = []
    thorax_files = []
    angio_files = []
    other_files = []

    thorax_terms = ["Thorax", "Thoracica", "Thx", "Abdomen", "Abd"]

    for filename in os.listdir(folder_path):
        if "Herz" in filename:
            herz_files.append(filename)
        elif any(term in filename for term in thorax_terms):
            thorax_files.append(filename)
        elif "Angio" in filename and not any(term in filename for term in thorax_terms):
            angio_files.append(filename)
        else:
            other_files.append(filename)

    return {
        "Herz": herz_files,
        "Thorax/Abdomen": thorax_files,
        "Angio": angio_files,
        "Other": other_files
    }

def generate_markdown(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# File Analysis Results\n\n")
        
        for category, files in results.items():
            f.write(f"## {category} Files\n\n")
            f.write(f"Number of files: {len(files)}\n\n")
            f.write("### Filenames:\n\n")
            for filename in files:
                f.write(f"- {filename}\n")
            f.write("\n")

if __name__ == "__main__":
    folder_path = input("Enter the folder path to analyze: ")
    output_file = "file_analysis_results.md"

    results = analyze_files(folder_path)
    generate_markdown(results, output_file)

    print(f"Analysis complete. Results saved to {output_file}")