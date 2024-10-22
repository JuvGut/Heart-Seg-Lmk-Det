import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import csv

sys.path.append("..")
sys.path.append(".")
from detection3d.vis.gen_images import gen_plane_images, load_coordinates_from_csv

def parse_arguments() -> argparse.Namespace:
    default_image_folder = '/home/juval.gutknecht/Projects/Data/Dataset012_aligned/imagesTs'
    default_label_folder = '/home/juval.gutknecht/Projects/Data/Dataset012_aligned/landmarksTs_csv'
    default_detection_folder = '/home/juval.gutknecht/Projects/Data/results/inference_results_aaa'
    default_resolution = [0.43, 0.3, 0.43] # [x, y, z] in mm
    default_contrast_range = None
    default_output_folder = os.path.join(default_detection_folder, 'report')
    default_generate_pictures = False

    parser = argparse.ArgumentParser(description='Analyze landmark detection results and generate report.')
    parser.add_argument('--image_folder', default=default_image_folder, type=str, help='Folder containing the source data.')
    parser.add_argument('--label_folder', default=default_label_folder, type=str, help='Folder with CSV files of labelled landmark coordinates.')
    parser.add_argument('--detection_folder', default=default_detection_folder, type=str, help='Folder with CSV files of detected landmark coordinates.')
    parser.add_argument('--resolution', type=float, nargs=3, default=default_resolution, help="Resolution of the snapshot images.")
    parser.add_argument('--contrast_range', default=default_contrast_range, type=float, nargs=2, help='Minimal and maximal value of contrast intensity window.')
    parser.add_argument('--output_folder', default=default_output_folder, type=str, help='Folder for generated output.')
    parser.add_argument('--generate_pictures', default=default_generate_pictures, action='store_true', help='Generate snapshot images of landmarks.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for step-by-step calculation output')

    return parser.parse_args()

def load_landmark_files(folder: str) -> List[str]:
    landmark_csvs = glob.glob(os.path.join(folder, "*.csv"))
    assert len(landmark_csvs) > 0, f"No CSV files found in {folder}"
    return sorted(landmark_csvs)

def load_landmarks(csv_files: List[str]) -> Dict[str, Dict[str, List[float]]]:
    landmarks = {}
    for csv_file in csv_files:
        file_name = os.path.basename(csv_file).split('_')[0]  # Get the BS-XXX part
        landmarks[file_name] = load_coordinates_from_csv(csv_file)
    return landmarks

def find_common_files(label_files: List[str], detection_files: List[str]) -> Tuple[List[str], List[str]]:
    label_basenames = [os.path.basename(f).split('_')[0] for f in label_files]
    detection_basenames = [os.path.basename(f).split('_')[0] for f in detection_files]
    common_basenames = set(label_basenames) & set(detection_basenames)
    
    common_label_files = [f for f in label_files if os.path.basename(f).split('_')[0] in common_basenames]
    common_detection_files = [f for f in detection_files if os.path.basename(f).split('_')[0] in common_basenames]
    
    return common_label_files, common_detection_files

def analyze_landmarks(label_landmarks: Dict[str, Dict[str, List[float]]], 
                      detection_landmarks: Dict[str, Dict[str, List[float]]] = None,
                      debug: bool = False) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    all_landmarks = set()
    for landmarks in [label_landmarks] + ([detection_landmarks] if detection_landmarks else []):
        all_landmarks.update(landmarks[list(landmarks.keys())[0]].keys())
    
    landmark_data = {landmark: {'detected': 0, 'total': 0, 'errors': []} for landmark in all_landmarks}
    case_data = {case: {'total': 0, 'detected': 0} for case in label_landmarks.keys()}
    detailed_errors = []

    if debug:
        print("Step 1: Collecting data and calculating errors")

    for case, case_landmarks in label_landmarks.items():
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
                        error = np.array(coords) - np.array(detection_coords)
                        error_magnitude = np.linalg.norm(error)
                        landmark_data[landmark]['errors'].append(error_magnitude)
                        
                        detailed_errors.append({
                            'image': case,
                            'landmark': landmark,
                            'x_detected': detection_coords[0],
                            'y_detected': detection_coords[1],
                            'z_detected': detection_coords[2],
                            'x_groundtruth': coords[0],
                            'y_groundtruth': coords[1],
                            'z_groundtruth': coords[2],
                            'x_error': error[0],
                            'y_error': error[1],
                            'z_error': error[2],
                            'error_magnitude': error_magnitude
                        })

                        if debug:
                            print(f"  Case: {case}, Landmark: {landmark}")
                            print(f"    Label coordinates: {coords}")
                            print(f"    Detection coordinates: {detection_coords}")
                            print(f"    Error: {error:.2f}")
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
                print(f"    Mean error: {data['mean_error']:.2f} mm")
                print(f"    Median error: {data['median_error']:.2f} mm")
                print(f"    Std error: {data['std_error']:.2f} mm")
                print(f"    Max error: {data['max_error']:.2f} mm")
                print(f"    25th percentile: {data['percentile_25']:.2f} mm")
                print(f"    75th percentile: {data['percentile_75']:.2f} mm")
                print(f"    90th percentile: {data['percentile_90']:.2f} mm")
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

    return landmark_data, case_data, detailed_errors

def generate_html_report(landmark_data: Dict[str, Dict[str, float]], case_data: Dict[str, Dict[str, float]], detailed_errors: List[Dict[str, any]], output_folder: str, compare_mode: bool):
    # Sort landmarks and cases by detection rate
    sorted_landmarks = sorted(landmark_data.items(), key=lambda x: x[1]['detection_rate'], reverse=True)
    sorted_cases = sorted(case_data.items(), key=lambda x: x[1]['detection_rate'], reverse=True)

    # Generate HTML content
    html_content = """
    <html>
    <head>
        <style>
            table, th, td {{
                border: 1px solid black;
                border-collapse: collapse;
                padding: 5px;
            }}
        </style>
    </head>
    <body>
        <h2>Landmark Detection Summary</h2>
        <table>
            <tr>
                <th>Landmark</th>
                <th>Detection Rate</th>
                <th>Detected</th>
                <th>Total</th>
                <th>Mean Error (mm)</th>
                <th>Median Error (mm)</th>
                <th>Std Error (mm)</th>
                <th>Max Error (mm)</th>
                <th>25th Percentile (mm)</th>
                <th>75th Percentile (mm)</th>
                <th>90th Percentile (mm)</th>
                <th>Inlier Rate (5mm)</th>
            </tr>
            {0}
        </table>

        <h2>Case Detection Summary</h2>
        <table>
            <tr>
                <th>Case</th>
                <th>Detection Rate</th>
                <th>Detected</th>
                <th>Total</th>
            </tr>
            {1}
        </table>

        <h2>Detailed Error Information</h2>
        <table>
            <tr>
                <th>Image</th>
                <th>Landmark</th>
                <th>X Detected</th>
                <th>Y Detected</th>
                <th>Z Detected</th>
                <th>X Ground Truth</th>
                <th>Y Ground Truth</th>
                <th>Z Ground Truth</th>
                <th>X Error</th>
                <th>Y Error</th>
                <th>Z Error</th>
                <th>Error Magnitude</th>
            </tr>
            {2}
        </table>
    </body>
    </html>
    """.format(
        "\n".join([
            "<tr><td>{0}</td><td>{1:.2f}%</td><td>{2}</td><td>{3}</td><td>{4}</td><td>{5}</td><td>{6}</td><td>{7}</td><td>{8}</td><td>{9}</td><td>{10}</td><td>{11}</td></tr>".format(
                landmark, data['detection_rate'], data['detected'], data['total'],
                f"{data['mean_error']:.2f}" if isinstance(data['mean_error'], float) else data['mean_error'],
                f"{data['median_error']:.2f}" if isinstance(data['median_error'], float) else data['median_error'],
                f"{data['std_error']:.2f}" if isinstance(data['std_error'], float) else data['std_error'],
                f"{data['max_error']:.2f}" if isinstance(data['max_error'], float) else data['max_error'],
                f"{data['percentile_25']:.2f}" if isinstance(data['percentile_25'], float) else data['percentile_25'],
                f"{data['percentile_75']:.2f}" if isinstance(data['percentile_75'], float) else data['percentile_75'],
                f"{data['percentile_90']:.2f}" if isinstance(data['percentile_90'], float) else data['percentile_90'],
                f"{data['inlier_rate_5mm']:.2f}%" if isinstance(data['inlier_rate_5mm'], float) else data['inlier_rate_5mm']
            ) for landmark, data in sorted_landmarks
        ]),
        "\n".join([
            "<tr><td>{0}</td><td>{1:.2f}%</td><td>{2}</td><td>{3}</td></tr>".format(
                case, data['detection_rate'], data['detected'], data['total']
            ) for case, data in sorted_cases
        ]),
        "\n".join([
            "<tr><td>{image}</td><td>{landmark}</td><td>{x_detected:.2f}</td><td>{y_detected:.2f}</td><td>{z_detected:.2f}</td><td>{x_groundtruth:.2f}</td><td>{y_groundtruth:.2f}</td><td>{z_groundtruth:.2f}</td><td>{x_error:.2f}</td><td>{y_error:.2f}</td><td>{z_error:.2f}</td><td>{error_magnitude:.2f}</td></tr>".format(**error)
            for error in detailed_errors
        ])
    )

    # Save the HTML report
    output_file = os.path.join(output_folder, 'landmark_detection_report.html')
    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"HTML report generated and saved to {output_file}")

    # Generate CSV report
    csv_content = "landmark_idx,landmark_name,detected,total,detection_rate (%),mean_error,median_error,std_error,max_error,percentile_25,percentile_75,percentile_90,inlier_rate_5mm\n"
    for idx, (landmark, data) in enumerate(sorted_landmarks):
        csv_content += f"{idx},{landmark},{data['detected']},{data['total']},{data['detection_rate']:.2f},"
        csv_content += f"{data['mean_error'] if isinstance(data['mean_error'], float) else 'N/A'},"
        csv_content += f"{data['median_error'] if isinstance(data['median_error'], float) else 'N/A'},"
        csv_content += f"{data['std_error'] if isinstance(data['std_error'], float) else 'N/A'},"
        csv_content += f"{data['max_error'] if isinstance(data['max_error'], float) else 'N/A'},"
        csv_content += f"{data['percentile_25'] if isinstance(data['percentile_25'], float) else 'N/A'},"
        csv_content += f"{data['percentile_75'] if isinstance(data['percentile_75'], float) else 'N/A'},"
        csv_content += f"{data['percentile_90'] if isinstance(data['percentile_90'], float) else 'N/A'},"
        csv_content += f"{data['inlier_rate_5mm'] if isinstance(data['inlier_rate_5mm'], float) else 'N/A'}\n"

    csv_output_file = os.path.join(output_folder, 'landmark_detection_report.csv')
    with open(csv_output_file, 'w') as f:
        f.write(csv_content)

    print(f"CSV report generated and saved to {csv_output_file}")

    # Generate CSV report for detailed errors
    detailed_csv_content = "image,landmark,x_detected,y_detected,z_detected,x_groundtruth,y_groundtruth,z_groundtruth,x_error,y_error,z_error,error_magnitude\n"
    for error in detailed_errors:
        detailed_csv_content += "{image},{landmark},{x_detected:.2f},{y_detected:.2f},{z_detected:.2f},{x_groundtruth:.2f},{y_groundtruth:.2f},{z_groundtruth:.2f},{x_error:.2f},{y_error:.2f},{z_error:.2f},{error_magnitude:.2f}\n".format(**error)

    detailed_csv_output_file = os.path.join(output_folder, 'landmark_detection_detailed.csv')
    with open(detailed_csv_output_file, 'w') as f:
        f.write(detailed_csv_content)

    print(f"CSV detailed report generated and saved to {detailed_csv_output_file}")

def generate_tableau_data(landmark_data: Dict[str, Dict[str, float]], case_data: Dict[str, Dict[str, float]], detailed_errors: List[Dict[str, any]], output_folder: str):
    # Prepare landmark summary data
    landmark_summary = []
    for landmark, data in landmark_data.items():
        landmark_summary.append({
            'Landmark': landmark,
            'Detection Rate (%)': data['detection_rate'],
            'Detected': data['detected'],
            'Total': data['total'],
            'Mean Error (mm)': data['mean_error'] if isinstance(data['mean_error'], float) else None,
            'Median Error (mm)': data['median_error'] if isinstance(data['median_error'], float) else None,
            'Std Error (mm)': data['std_error'] if isinstance(data['std_error'], float) else None,
            'Max Error (mm)': data['max_error'] if isinstance(data['max_error'], float) else None,
            'Percentile 25 (mm)': data['percentile_25'] if isinstance(data['percentile_25'], float) else None,
            'Percentile 75 (mm)': data['percentile_75'] if isinstance(data['percentile_75'], float) else None,
            'Percentile 90 (mm)': data['percentile_90'] if isinstance(data['percentile_90'], float) else None,
            'Inlier Rate (5mm) (%)': data['inlier_rate_5mm'] if isinstance(data['inlier_rate_5mm'], float) else None
        })
    
    # Prepare case summary data
    case_summary = []
    for case, data in case_data.items():
        case_summary.append({
            'Case': case,
            'Detection Rate (%)': data['detection_rate'],
            'Detected': data['detected'],
            'Total': data['total']
        })
    
    # Convert to pandas DataFrames
    landmark_df = pd.DataFrame(landmark_summary)
    case_df = pd.DataFrame(case_summary)
    detailed_errors_df = pd.DataFrame(detailed_errors)
    
    # Save as CSV files
    landmark_df.to_csv(os.path.join(output_folder, 'tableau_landmark_summary.csv'), index=False)
    case_df.to_csv(os.path.join(output_folder, 'tableau_case_summary.csv'), index=False)
    detailed_errors_df.to_csv(os.path.join(output_folder, 'tableau_detailed_errors.csv'), index=False)
    
    print(f"Tableau-ready CSV files generated in {output_folder}")

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
    
    landmark_data, case_data, detailed_errors = analyze_landmarks(label_landmarks, detection_landmarks, args.debug)
    generate_html_report(landmark_data, case_data, detailed_errors, args.output_folder, compare_mode)
    generate_tableau_data(landmark_data, case_data, detailed_errors, args.output_folder)

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