import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append("..")
sys.path.append(".")
from detection3d.vis.gen_images import gen_plane_images, load_coordinates_from_csv

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the landmark analysis script.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Analyze landmark detection results and generate report.')
    parser.add_argument('--image_folder', required=True, type=str, help='Folder containing the source data.')
    parser.add_argument('--label_folder', required=True, type=str, help='Folder with CSV files of labelled landmark coordinates.')
    parser.add_argument('--detection_folder', type=str, help='Folder with CSV files of detected landmark coordinates.')
    parser.add_argument('--resolution', type=float, nargs=3, default=[0.43, 0.3, 0.43], help="Resolution of the snapshot images in mm/voxel [x, y, z].")
    parser.add_argument('--contrast_range', type=float, nargs=2, help='Minimal and maximal value of contrast intensity window.')
    parser.add_argument('--output_folder', required=True, type=str, help='Folder for generated output.')
    parser.add_argument('--generate_pictures', action='store_true', help='Generate snapshot images of landmarks.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for step-by-step calculation output')
    return parser.parse_args()

def load_landmark_files(folder: str) -> List[str]:
    """
    Load landmark CSV files from the specified folder.
    
    Args:
        folder (str): Path to the folder containing landmark CSV files
    
    Returns:
        List[str]: Sorted list of CSV file paths
    """
    landmark_csvs = glob.glob(os.path.join(folder, "*.csv"))
    assert len(landmark_csvs) > 0, f"No CSV files found in {folder}"
    return sorted(landmark_csvs)

def load_landmarks(csv_files: List[str]) -> Dict[str, Dict[str, List[float]]]:
    """
    Load landmark coordinates from CSV files.
    
    Args:
        csv_files (List[str]): List of CSV file paths
    
    Returns:
        Dict[str, Dict[str, List[float]]]: Nested dictionary of landmark coordinates
    """
    landmarks = {}
    for csv_file in tqdm(csv_files, desc="Loading landmarks"):
        file_name = os.path.basename(csv_file).split('_')[0]
        landmarks[file_name] = load_coordinates_from_csv(csv_file)
    return landmarks

def find_common_files(label_files: List[str], detection_files: List[str]) -> Tuple[List[str], List[str]]:
    """
    Find common files between label and detection datasets.
    
    Args:
        label_files (List[str]): List of label file paths
        detection_files (List[str]): List of detection file paths
    
    Returns:
        Tuple[List[str], List[str]]: Common label files and detection files
    """
    label_basenames = set(os.path.basename(f).split('_')[0] for f in label_files)
    detection_basenames = set(os.path.basename(f).split('_')[0] for f in detection_files)
    common_basenames = label_basenames & detection_basenames
    
    common_label_files = [f for f in label_files if os.path.basename(f).split('_')[0] in common_basenames]
    common_detection_files = [f for f in detection_files if os.path.basename(f).split('_')[0] in common_basenames]
    
    return common_label_files, common_detection_files

def analyze_landmarks(label_landmarks: Dict[str, Dict[str, List[float]]], 
                      detection_landmarks: Dict[str, Dict[str, List[float]]] = None,
                      debug: bool = False) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Analyze landmark detection results and calculate metrics.
    
    Args:
        label_landmarks (Dict[str, Dict[str, List[float]]]): Ground truth landmark coordinates
        detection_landmarks (Dict[str, Dict[str, List[float]]], optional): Detected landmark coordinates
        debug (bool, optional): Enable debug output
    
    Returns:
        Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]: Landmark and case analysis results
    """
    all_landmarks = set()
    for landmarks in [label_landmarks] + ([detection_landmarks] if detection_landmarks else []):
        all_landmarks.update(landmarks[next(iter(landmarks))].keys())
    
    landmark_data = {landmark: {'detected': 0, 'total': 0, 'errors': []} for landmark in all_landmarks}
    case_data = {case: {'total': 0, 'detected': 0} for case in label_landmarks.keys()}

    if debug:
        print("Step 1: Collecting data and calculating errors")

    for case, case_landmarks in tqdm(label_landmarks.items(), desc="Analyzing landmarks"):
        case_data[case]['total'] = len(case_landmarks)
        case_detected = 0
        for landmark, coords in case_landmarks.items():
            landmark_data[landmark]['total'] += 1
            
            if detection_landmarks:
                detection_coords = detection_landmarks.get(case, {}).get(landmark, [-1, -1, -1])
                if not np.allclose(detection_coords, [-1, -1, -1]):
                    landmark_data[landmark]['detected'] += 1
                    case_detected += 1
                    
                    if not np.allclose(coords, [-1, -1, -1]):
                        error = np.linalg.norm(np.array(coords) - np.array(detection_coords))
                        landmark_data[landmark]['errors'].append(error)
                        
                        if debug:
                            print(f"  Case: {case}, Landmark: {landmark}")
                            print(f"    Label coordinates (mm): {coords}")
                            print(f"    Detection coordinates (mm): {detection_coords}")
                            print(f"    Error (mm): {error:.2f}")
            else:
                if not np.allclose(coords, [-1, -1, -1]):
                    landmark_data[landmark]['detected'] += 1
                    case_detected += 1
        
        case_data[case]['detected'] = case_detected

    if debug:
        print("\nStep 2: Calculating metrics for each landmark")

    for landmark, data in landmark_data.items():
        data['detection_rate'] = (data['detected'] / data['total']) * 100
        if data['errors']:
            data['mean_error'] = np.mean(data['errors'])
            data['median_error'] = np.median(data['errors'])
            data['std_error'] = np.std(data['errors'])
            data['max_error'] = np.max(data['errors'])
            data['percentile_25'] = np.percentile(data['errors'], 25)
            data['percentile_75'] = np.percentile(data['errors'], 75)
            data['percentile_90'] = np.percentile(data['errors'], 90)
            data['inlier_rate_5mm'] = np.mean(np.array(data['errors']) < 5) * 100
            
            if debug:
                print(f"\n  Landmark: {landmark}")
                print(f"    Total occurrences: {data['total']}")
                print(f"    Detected: {data['detected']}")
                print(f"    Detection rate: {data['detection_rate']:.2f}%")
                print(f"    Number of valid error measurements: {len(data['errors'])}")
                print(f"    Mean error (mm): {data['mean_error']:.2f}")
                print(f"    Median error (mm): {data['median_error']:.2f}")
                print(f"    Std error (mm): {data['std_error']:.2f}")
                print(f"    Max error (mm): {data['max_error']:.2f}")
                print(f"    25th percentile (mm): {data['percentile_25']:.2f}")
                print(f"    75th percentile (mm): {data['percentile_75']:.2f}")
                print(f"    90th percentile (mm): {data['percentile_90']:.2f}")
                print(f"    Inlier rate (5mm): {data['inlier_rate_5mm']:.2f}%")
        else:
            data['mean_error'] = data['median_error'] = data['std_error'] = data['max_error'] = 'N/A'
            data['percentile_25'] = data['percentile_75'] = data['percentile_90'] = 'N/A'
            data['inlier_rate_5mm'] = 'N/A'
            
            if debug:
                print(f"\n  Landmark: {landmark}")
                print(f"    Total occurrences: {data['total']}")
                print(f"    Detected: {data['detected']}")
                print(f"    Detection rate: {data['detection_rate']:.2f}%")
                print("    No valid error measurements available")

    if debug:
        print("\nStep 3: Calculating case detection rates")

    for case in case_data:
        case_data[case]['detection_rate'] = (case_data[case]['detected'] / case_data[case]['total']) * 100
        if debug:
            print(f"  Case: {case}")
            print(f"    Total landmarks: {case_data[case]['total']}")
            print(f"    Detected landmarks: {case_data[case]['detected']}")
            print(f"    Detection rate: {case_data[case]['detection_rate']:.2f}%")

    return landmark_data, case_data

def generate_reports(landmark_data: Dict[str, Dict[str, float]], case_data: Dict[str, Dict[str, float]], output_folder: str, compare_mode: bool):
    """
    Generate HTML and CSV reports for landmark analysis results.
    
    Args:
        landmark_data (Dict[str, Dict[str, float]]): Landmark analysis results
        case_data (Dict[str, Dict[str, float]]): Case analysis results
        output_folder (str): Output folder for reports
        compare_mode (bool): True if comparing label and detection data
    """
    # Sort landmarks and cases by detection rate
    sorted_landmarks = sorted(landmark_data.items(), key=lambda x: x[1]['detection_rate'], reverse=True)
    sorted_cases = sorted(case_data.items(), key=lambda x: x[1]['detection_rate'], reverse=True)

    # Generate HTML report
    html_content = generate_html_content(sorted_landmarks, sorted_cases, compare_mode)
    html_output_file = os.path.join(output_folder, 'landmark_detection_report.html')
    with open(html_output_file, 'w') as f:
        f.write(html_content)
    print(f"HTML report generated and saved to {html_output_file}")

    # Generate CSV report
    csv_content = generate_csv_content(sorted_landmarks)
    csv_output_file = os.path.join(output_folder, 'landmark_detection_report.csv')
    with open(csv_output_file, 'w') as f:
        f.write(csv_content)
    print(f"CSV report generated and saved to {csv_output_file}")

    # Generate summary plots
    generate_summary_plots(landmark_data, case_data, output_folder)

def generate_html_content(sorted_landmarks, sorted_cases, compare_mode):
    """Generate HTML content for the report."""
    html_content = """
    <html>
    <head>
        <style>
            table, th, td {{
                border: 1px solid black;
                border-collapse: collapse;
                padding: 5px;
            }}
            .summary {{
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>Landmark Detection Analysis Report</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>Total landmarks analyzed: {total_landmarks}</p>
            <p>Average detection rate: {avg_detection_rate:.2f}%</p>
            <p>Median detection rate: {median_detection_rate:.2f}%</p>
        </div>
        
        <h2>Landmark Detection Summary</h2>
        <table>
            <tr>
                <th>Landmark</th>
                <th>Detection Rate (%)</th>
                <th>Detected</th>
                <th>Total</th>
                <th>Mean Error (mm)</th>
                <th>Median Error (mm)</th>
                <th>Std Error (mm)</th>
                <th>Max Error (mm)</th>
                <th>25th Percentile (mm)</th>
                <th>75th Percentile (mm)</th>
                <th>90th Percentile (mm)</th>
                <th>Inlier Rate (5mm) (%)</th>
            </tr>
            {landmark_rows}
        </table>

        <h2>Case Detection Summary</h2>
        <table>
            <tr>
                <th>Case</th>
                <th>Detection Rate (%)</th>
                <th>Detected</th>
                <th>Total</th>
            </tr>
            {case_rows}
        </table>
    </body>
    </html>
    """
    
    total_landmarks = len(sorted_landmarks)
    avg_detection_rate = np.mean([data['detection_rate'] for _, data in sorted_landmarks])
    median_detection_rate = np.median([data['detection_rate'] for _, data in sorted_landmarks])
    
    def format_value(value):
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)
    
    landmark_rows = []
    for landmark, data in sorted_landmarks:
        row = f"""<tr>
            <td>{landmark}</td>
            <td>{format_value(data['detection_rate'])}</td>
            <td>{data['detected']}</td>
            <td>{data['total']}</td>
            <td>{format_value(data['mean_error'])}</td>
            <td>{format_value(data['median_error'])}</td>
            <td>{format_value(data['std_error'])}</td>
            <td>{format_value(data['max_error'])}</td>
            <td>{format_value(data['percentile_25'])}</td>
            <td>{format_value(data['percentile_75'])}</td>
            <td>{format_value(data['percentile_90'])}</td>
            <td>{format_value(data['inlier_rate_5mm'])}</td>
        </tr>"""
        landmark_rows.append(row)
    
    case_rows = []
    for case, data in sorted_cases:
        row = f"""<tr>
            <td>{case}</td>
            <td>{format_value(data['detection_rate'])}</td>
            <td>{data['detected']}</td>
            <td>{data['total']}</td>
        </tr>"""
        case_rows.append(row)
    
    return html_content.format(
        total_landmarks=total_landmarks,
        avg_detection_rate=avg_detection_rate,
        median_detection_rate=median_detection_rate,
        landmark_rows="\n".join(landmark_rows),
        case_rows="\n".join(case_rows)
    )

def generate_csv_content(sorted_landmarks):
    """Generate CSV content for the report."""
    csv_content = "landmark_idx,landmark_name,detected,total,detection_rate (%),mean_error (mm),median_error (mm),std_error (mm),max_error (mm),percentile_25 (mm),percentile_75 (mm),percentile_90 (mm),inlier_rate_5mm (%)\n"
    
    def format_value(value):
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)
    
    for idx, (landmark, data) in enumerate(sorted_landmarks):
        row = [
            str(idx),
            landmark,
            str(data['detected']),
            str(data['total']),
            format_value(data['detection_rate']),
            format_value(data['mean_error']),
            format_value(data['median_error']),
            format_value(data['std_error']),
            format_value(data['max_error']),
            format_value(data['percentile_25']),
            format_value(data['percentile_75']),
            format_value(data['percentile_90']),
            format_value(data['inlier_rate_5mm'])
        ]
        csv_content += ",".join(row) + "\n"
    
    return csv_content

def generate_summary_plots(landmark_data, case_data, output_folder):
    """Generate summary plots for the analysis results."""
    # Detection rate distribution
    plt.figure(figsize=(10, 6))
    detection_rates = [data['detection_rate'] for data in landmark_data.values()]
    plt.hist(detection_rates, bins=20, edgecolor='black')
    plt.title('Distribution of Landmark Detection Rates')
    plt.xlabel('Detection Rate (%)')
    plt.ylabel('Number of Landmarks')
    plt.savefig(os.path.join(output_folder, 'detection_rate_distribution.png'))
    plt.close()

    # Error distribution (for landmarks with error data)
    errors = [error for data in landmark_data.values() for error in data.get('errors', [])]
    if errors:
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, edgecolor='black')
        plt.title('Distribution of Landmark Detection Errors')
        plt.xlabel('Error (mm)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_folder, 'error_distribution.png'))
        plt.close()

    # Case detection rate distribution
    plt.figure(figsize=(10, 6))
    case_detection_rates = [data['detection_rate'] for data in case_data.values()]
    plt.hist(case_detection_rates, bins=20, edgecolor='black')
    plt.title('Distribution of Case Detection Rates')
    plt.xlabel('Detection Rate (%)')
    plt.ylabel('Number of Cases')
    plt.savefig(os.path.join(output_folder, 'case_detection_rate_distribution.png'))
    plt.close()

def main():
    args = parse_arguments()
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    label_files = load_landmark_files(args.label_folder)
    label_landmarks = load_landmarks(label_files)
    
    if args.detection_folder:
        detection_files = load_landmark_files(args.detection_folder)
        common_label_files, common_detection_files = find_common_files(label_files, detection_files)
        label_landmarks = load_landmarks(common_label_files)
        detection_landmarks = load_landmarks(common_detection_files)
        compare_mode = True
    else:
        detection_landmarks = None
        compare_mode = False
    
    landmark_data, case_data = analyze_landmarks(label_landmarks, detection_landmarks, args.debug)
    generate_reports(landmark_data, case_data, args.output_folder, compare_mode)
    
    if args.generate_pictures:
        print('Generating planes for the labelled landmarks.')
        gen_plane_images(args.image_folder, label_landmarks, 'label',
                         args.contrast_range, args.resolution, args.output_folder)
        
        if compare_mode:
            print('Generating planes for the detected landmarks.')
            gen_plane_images(args.image_folder, detection_landmarks, 'detection',
                             args.contrast_range, args.resolution, args.output_folder)

if __name__ == '__main__':
    main()


# python scripts/gen_html_report_v3.py --image_folder /home/juval.gutknecht/Projects/Data/Dataset012_aligned/imagesTs --label_folder /home/juval.gutknecht/Projects/Data/Dataset012_aligned/landmarksTs_csv --detection_folder /home/juval.gutknecht/Projects/Data/results/inference_model_comparison/newmask_012_202020  --output_folder /home/juval.gutknecht/Projects/Data/results/inference_model_comparison/newmask_012_202020/report 