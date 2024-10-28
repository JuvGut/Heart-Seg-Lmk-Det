import argparse
import glob
import os
import sys
import json
import pathlib as Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import SimpleITK as sitk

sys.path.append("..")
sys.path.append(".")
from detection3d.vis.gen_images import gen_plane_images, load_coordinates_from_csv

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the landmark analysis script."""
    parser = argparse.ArgumentParser(description='Analyze landmark detection results and generate report.')
    parser.add_argument('--image_folder', required=True, type=str, help='Folder containing the source data.')
    parser.add_argument('--label_folder', required=True, type=str, help='Folder with CSV files of labelled landmark coordinates.')
    parser.add_argument('--detection_folder', required=True, type=str, help='Folder with CSV files of detected landmark coordinates.')
    parser.add_argument('--output_folder', required=True, type=str, help='Folder for generated output.')
    parser.add_argument('--contrast_range', type=float, nargs=2, help='Minimal and maximal value of contrast intensity window.')
    parser.add_argument('--generate_pictures', action='store_true', help='Generate snapshot images of landmarks.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for step-by-step calculation output')
    return parser.parse_args()

def extract_model_name(detection_folder: str) -> str:
    """
    Extract model name from the detection folder path.
    
    Args:
        detection_folder (str): Path to the detection folder
        
    Returns:
        str: Extracted model name
    """
    # Get the last directory name from the path
    model_name = os.path.basename(detection_folder.rstrip('/'))
    
    # If the last directory is 'report', use the parent directory
    if model_name == 'report':
        model_name = os.path.basename(os.path.dirname(detection_folder.rstrip('/')))
    
    return model_name

def get_image_resolution(image_path: str) -> List[float]:
    """
    Get the resolution (spacing) of a medical image.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        List[float]: Resolution [x, y, z] in mm/voxel
    """
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(image_path)
        reader.ReadImageInformation()
        spacing = reader.GetSpacing()
        return list(spacing)  # [x, y, z] in mm/voxel
    except Exception as e:
        print(f"Warning: Could not read resolution from {image_path}: {e}")
        return [1.0, 1.0, 1.0]  # Default to 1mm isotropic if reading fails

def load_landmark_files(folder: str) -> List[str]:
    """Load landmark CSV files from the specified folder."""
    landmark_csvs = glob.glob(os.path.join(folder, "*.csv"))
    assert len(landmark_csvs) > 0, f"No CSV files found in {folder}"
    return sorted(landmark_csvs)

def load_landmarks(csv_files: List[str]) -> Dict[str, Dict[str, List[float]]]:
    """Load landmark coordinates from CSV files."""
    landmarks = {}
    for csv_file in tqdm(csv_files, desc="Loading landmarks"):
        file_name = os.path.basename(csv_file).split('_')[0]
        landmarks[file_name] = load_coordinates_from_csv(csv_file)
    return landmarks

def find_image_file(image_folder: str, case: str) -> Optional[str]:
    """
    Find image file that starts with the case identifier and ends with _0000.nrrd
    
    Args:
        image_folder (str): Path to image directory
        case (str): Case identifier (e.g., BS-186)
        
    Returns:
        Optional[str]: Path to image file if found, None otherwise
    """
    # List all files in the directory
    try:
        files = os.listdir(image_folder)
        # Find files that start with the case ID and end with _0000.nrrd
        matching_files = [f for f in files if f.startswith(case) and f.endswith('_0000.nrrd')]
        if matching_files:
            return os.path.join(image_folder, matching_files[0])
        
        # If not found, try with underscore instead of dash
        case_underscore = case.replace('-', '_')
        matching_files = [f for f in files if f.startswith(case_underscore) and f.endswith('_0000.nrrd')]
        if matching_files:
            return os.path.join(image_folder, matching_files[0])
            
        if not matching_files:
            print(f"Debug: No files found starting with {case} or {case_underscore}")
            print(f"Debug: Available files: {[f for f in files if f.endswith('_0000.nrrd')]}")
            
        return None
            
    except Exception as e:
        print(f"Error accessing directory {image_folder}: {e}")
        return None

def find_common_files(label_files: List[str], detection_files: List[str]) -> Tuple[List[str], List[str]]:
    """Find common files between label and detection datasets."""
    label_basenames = set(os.path.basename(f).split('_')[0] for f in label_files)
    detection_basenames = set(os.path.basename(f).split('_')[0] for f in detection_files)
    common_basenames = label_basenames & detection_basenames
    
    common_label_files = [f for f in label_files if os.path.basename(f).split('_')[0] in common_basenames]
    common_detection_files = [f for f in detection_files if os.path.basename(f).split('_')[0] in common_basenames]
    
    return sorted(common_label_files), sorted(common_detection_files)

def calculate_physical_error(coords1: List[float], coords2: List[float], resolution: List[float]) -> Tuple[float, List[float]]:
    """
    Calculate physical error between two coordinate points considering image resolution.
    
    Args:
        coords1 (List[float]): First coordinate point [x, y, z]
        coords2 (List[float]): Second coordinate point [x, y, z]
        resolution (List[float]): Image resolution [x, y, z] in mm/voxel
        
    Returns:
        Tuple[float, List[float]]: Total error in mm and component-wise errors [x, y, z] in mm
    """
    error_voxels = np.array(coords1) - np.array(coords2)
    error_mm = error_voxels * np.array(resolution)
    return np.linalg.norm(error_mm), error_mm.tolist()

def analyze_landmarks(label_landmarks: Dict[str, Dict[str, List[float]]], 
                     detection_landmarks: Optional[Dict[str, Dict[str, List[float]]]], 
                     image_folder: str,
                     debug: bool = False) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], List[Dict]]:
    """
    Analyze landmark detection results and calculate metrics considering image resolution.
    """
    all_landmarks = set()
    for landmarks in [label_landmarks] + ([detection_landmarks] if detection_landmarks else []):
        all_landmarks.update(landmarks[next(iter(landmarks))].keys())
    
    landmark_data = {landmark: {'detected': 0, 'total': 0, 'errors': [], 'errors_by_axis': {'x': [], 'y': [], 'z': []}} 
                    for landmark in all_landmarks}
    case_data = {case: {'total': 0, 'detected': 0, 'resolution': None} for case in label_landmarks.keys()}  # Added resolution field
    detailed_errors = []

    if debug:
        print("Step 1: Collecting data and calculating errors")

    for case, case_landmarks in tqdm(label_landmarks.items(), desc="Analyzing landmarks"):
        # Get image resolution for this case with proper error handling
        try:
            image_path = find_image_file(image_folder, case)
            if image_path is None:
                raise FileNotFoundError(f"No image file found for case {case}")
            resolution = get_image_resolution(image_path)
            case_data[case]['resolution'] = resolution  # Store resolution in case_data
        except Exception as e:
            print(f"Warning: Could not get resolution for case {case}: {e}")
            resolution = [1.0, 1.0, 1.0]  # Default resolution if image not found
            case_data[case]['resolution'] = resolution
        
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
                        error_mm, error_components = calculate_physical_error(coords, detection_coords, resolution)
                        landmark_data[landmark]['errors'].append(error_mm)
                        landmark_data[landmark]['errors_by_axis']['x'].append(abs(error_components[0]))
                        landmark_data[landmark]['errors_by_axis']['y'].append(abs(error_components[1]))
                        landmark_data[landmark]['errors_by_axis']['z'].append(abs(error_components[2]))
                        
                        detailed_errors.append({
                            'image': case,
                            'landmark': landmark,
                            'resolution_x': resolution[0],
                            'resolution_y': resolution[1],
                            'resolution_z': resolution[2],
                            'x_detected': detection_coords[0],
                            'y_detected': detection_coords[1],
                            'z_detected': detection_coords[2],
                            'x_groundtruth': coords[0],
                            'y_groundtruth': coords[1],
                            'z_groundtruth': coords[2],
                            'x_error_mm': error_components[0],
                            'y_error_mm': error_components[1],
                            'z_error_mm': error_components[2],
                            'error_magnitude_mm': error_mm,
                            'x_error_voxels': detection_coords[0] - coords[0],  # Added voxel errors
                            'y_error_voxels': detection_coords[1] - coords[1],
                            'z_error_voxels': detection_coords[2] - coords[2]
                        })
                        
                        if debug:
                            print(f"\nCase: {case}, Landmark: {landmark}")
                            print(f"Resolution (mm/voxel): {resolution}")
                            print(f"Label coordinates (voxels): {coords}")
                            print(f"Detection coordinates (voxels): {detection_coords}")
                            print(f"Error (mm): {error_mm:.2f}")
                            print(f"Component errors (mm): x={error_components[0]:.2f}, "
                                  f"y={error_components[1]:.2f}, z={error_components[2]:.2f}")
            else:
                if not np.allclose(coords, [-1, -1, -1]):
                    landmark_data[landmark]['detected'] += 1
                    case_detected += 1
        
        case_data[case]['detected'] = case_detected

    # Calculate statistics for each landmark
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
            
            # Calculate axis-specific statistics
            for axis in ['x', 'y', 'z']:
                axis_errors = data['errors_by_axis'][axis]
                data[f'mean_error_{axis}'] = np.mean(axis_errors)
                data[f'median_error_{axis}'] = np.median(axis_errors)
                data[f'max_error_{axis}'] = np.max(axis_errors)
                data[f'std_error_{axis}'] = np.std(axis_errors)  # Added std for each axis
        else:
            data['mean_error'] = data['median_error'] = data['std_error'] = data['max_error'] = 'N/A'
            data['percentile_25'] = data['percentile_75'] = data['percentile_90'] = 'N/A'
            data['inlier_rate_5mm'] = 'N/A'
            for axis in ['x', 'y', 'z']:
                data[f'mean_error_{axis}'] = data[f'median_error_{axis}'] = data[f'max_error_{axis}'] = data[f'std_error_{axis}'] = 'N/A'

    # Calculate case detection rates
    for case in case_data:
        case_data[case]['detection_rate'] = (case_data[case]['detected'] / case_data[case]['total']) * 100

    return landmark_data, case_data, detailed_errors

def generate_html_report(landmark_data: Dict[str, Dict[str, float]], 
                        case_data: Dict[str, Dict[str, float]], 
                        detailed_errors: List[Dict],
                        output_folder: str,
                        model_name: str):
    """Generate comprehensive HTML report with enhanced error analysis."""
    # Sort landmarks and cases by detection rate
    sorted_landmarks = sorted(landmark_data.items(), key=lambda x: x[1]['detection_rate'], reverse=True)
    sorted_cases = sorted(case_data.items(), key=lambda x: x[1]['detection_rate'], reverse=True)

    # Calculate summary statistics
    total_landmarks = len(sorted_landmarks)
    avg_detection_rate = np.mean([data['detection_rate'] for _, data in sorted_landmarks])
    median_detection_rate = np.median([data['detection_rate'] for _, data in sorted_landmarks])
    
    # Calculate overall error statistics
    all_errors = [error['error_magnitude_mm'] for error in detailed_errors]
    if all_errors:
        overall_mean_error = np.mean(all_errors)
        overall_median_error = np.median(all_errors)
        overall_std_error = np.std(all_errors)
        inlier_rate_5mm = np.mean(np.array(all_errors) < 5) * 100
    else:
        overall_mean_error = overall_median_error = overall_std_error = inlier_rate_5mm = 'N/A'

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f5f5f5; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .summary-box {{
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 15px;
                margin-bottom: 20px;
            }}
            .error-distribution {{
                margin: 20px 0;
            }}
            .axis-specific {{
                margin: 20px 0;
                padding: 10px;
                background-color: #fff;
                border: 1px solid #e9ecef;
            }}
            h2 {{ color: #333; margin-top: 30px; }}
            .metric {{ margin: 10px 0; }}
            .highlight {{ color: #007bff; }}
            .model-name {{ 
                background-color: #e9ecef;
                padding: 10px;
                border-radius: 4px;
                margin-bottom: 20px;
                font-size: 1.2em;
                color: #495057;
            }}
            .info-icon {{
                display: inline-block;
                width: 16px;
                height: 16px;
                background-color: #6c757d;
                color: white;
                border-radius: 50%;
                text-align: center;
                line-height: 16px;
                font-size: 12px;
                cursor: pointer;
                margin-left: 5px;
            }}
            .explanation {{
                display: none;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                padding: 10px;
                margin: 5px 0;
                border-radius: 4px;
            }}
            .graph-section {{
                margin: 30px 0;
                padding: 20px;
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }}
            .graph-container {{
                margin: 20px 0;
            }}
            .graph-title {{
                font-weight: bold;
                margin-bottom: 10px;
            }}
            .graph-description {{
                color: #666;
                margin-bottom: 15px;
            }}
            .expandable-section {{
                border: 1px solid #dee2e6;
                border-radius: 4px;
                margin: 10px 0;
            }}
            .expandable-header {{
                background-color: #f8f9fa;
                padding: 10px;
                cursor: pointer;
                font-weight: bold;
            }}
            .expandable-content {{
                display: none;
                padding: 15px;
            }}
        </style>
        <script>
            function toggleExplanation(id) {{
                const element = document.getElementById(id);
                element.style.display = element.style.display === 'none' ? 'block' : 'none';
            }}
            
            function toggleSection(id) {{
                const element = document.getElementById(id);
                const content = element.querySelector('.expandable-content');
                content.style.display = content.style.display === 'none' ? 'block' : 'none';
            }}
        </script>
    </head>
    <body>
        <h1>Landmark Detection Analysis Report</h1>
        <div class="model-name">Model: {model_name}</div>

        <div class="summary-box">
            <h2>Overall Summary <span class="info-icon" onclick="toggleExplanation('summary-explanation')">?</span></h2>
            <div id="summary-explanation" class="explanation">
                This section provides key metrics summarizing the model's performance across all landmarks and cases.
            </div>
            <div class="metric">Total Landmarks Analyzed: <span class="highlight">{total_landmarks}</span></div>
            <div class="metric">Average Detection Rate: <span class="highlight">{avg_detection_rate:.2f}%</span></div>
            <div class="metric">Median Detection Rate: <span class="highlight">{median_detection_rate:.2f}%</span></div>
            <div class="metric">Mean Error: <span class="highlight">{overall_mean_error:.2f} mm</span></div>
            <div class="metric">Median Error: <span class="highlight">{overall_median_error:.2f} mm</span></div>
            <div class="metric">Standard Deviation: <span class="highlight">{overall_std_error:.2f} mm</span></div>
            <div class="metric">5mm Inlier Rate: <span class="highlight">{inlier_rate_5mm:.2f}%</span></div>
        </div>

        <div class="graph-section">
            <h2>Visualization Plots <span class="info-icon" onclick="toggleExplanation('plots-explanation')">?</span></h2>
            <div id="plots-explanation" class="explanation">
                Visual representations of error distributions and model performance metrics.
            </div>

            <div class="graph-container">
                <div class="graph-title">Error Distribution</div>
                <div class="graph-description">Distribution of landmark detection errors across all cases</div>
                <img src="error_distribution.png" alt="Error Distribution" style="width: 100%; max-width: 800px;">
            </div>

            <div class="graph-container">
                <div class="graph-title">Axis-specific Error Distribution</div>
                <div class="graph-description">Error distributions broken down by anatomical axis (X, Y, Z)</div>
                <img src="axis_error_distribution.png" alt="Axis-specific Error Distribution" style="width: 100%; max-width: 800px;">
            </div>

            <div class="graph-container">
                <div class="graph-title">Detection Rate Distribution</div>
                <div class="graph-description">Distribution of detection rates across landmarks</div>
                <img src="detection_rate_distribution.png" alt="Detection Rate Distribution" style="width: 100%; max-width: 800px;">
            </div>

            <div class="graph-container">
                <div class="graph-title">Error Boxplot</div>
                <div class="graph-description">Box plot showing error distributions by axis</div>
                <img src="error_boxplot.png" alt="Error Boxplot" style="width: 100%; max-width: 800px;">
            </div>

            <div class="graph-container">
                <div class="graph-title">Resolution Distribution</div>
                <div class="graph-description">Distribution of image resolutions across dataset</div>
                <img src="resolution_distribution.png" alt="Resolution Distribution" style="width: 100%; max-width: 800px;">
            </div>

            <div class="graph-container">
                <div class="graph-title">Error vs Resolution</div>
                <div class="graph-description">Relationship between detection errors and image resolution</div>
                <img src="error_vs_resolution.png" alt="Error vs Resolution" style="width: 100%; max-width: 800px;">
            </div>

            <div class="graph-container">
                <div class="graph-title">Top 10 Landmarks by Error</div>
                <div class="graph-description">Landmarks with the highest mean detection errors</div>
                <img src="top_10_errors.png" alt="Top 10 Errors" style="width: 100%; max-width: 800px;">
            </div>
        </div>

        <div class="expandable-section" id="landmark-section">
            <div class="expandable-header" onclick="toggleSection('landmark-section')">
                Landmark Detection Summary
                <span class="info-icon" onclick="toggleExplanation('landmark-explanation'); event.stopPropagation()">?</span>
            </div>
            <div id="landmark-explanation" class="explanation">
                Detailed performance metrics for each individual landmark, including detection rates and error statistics.
            </div>
            <div class="expandable-content">
                <table>
                    <tr>
                        <th>Landmark</th>
                        <th>Detection Rate (%)</th>
                        <th>Detected/Total</th>
                        <th>Mean Error (mm)</th>
                        <th>X Error (mm)</th>
                        <th>Y Error (mm)</th>
                        <th>Z Error (mm)</th>
                        <th>Median Error (mm)</th>
                        <th>Max Error (mm)</th>
                        <th>5mm Inlier Rate (%)</th>
                    </tr>
                    {landmark_rows}
                </table>
            </div>
        </div>

        <div class="expandable-section" id="case-section">
            <div class="expandable-header" onclick="toggleSection('case-section')">
                Case Detection Summary
                <span class="info-icon" onclick="toggleExplanation('case-explanation'); event.stopPropagation()">?</span>
            </div>
            <div id="case-explanation" class="explanation">
                Per-case detection statistics and image spacing information.
            </div>
            <div class="expandable-content">
                <table>
                    <tr>
                        <th>Case</th>
                        <th>Detection Rate (%)</th>
                        <th>Detected/Total</th>
                        <th>Image Spacing (mm)</th>
                    </tr>
                    {case_rows}
                </table>
            </div>
        </div>

    </body>
    </html>
    """

    def format_value(value, precision=2):
        """Helper function to format values with proper handling of N/A"""
        if isinstance(value, (int, float)):
            return f"{value:.{precision}f}"
        return 'N/A'

    # Generate landmark rows
    landmark_rows = []
    for landmark, data in sorted_landmarks:
        row = f"""
        <tr>
            <td>{landmark}</td>
            <td>{format_value(data['detection_rate'])}%</td>
            <td>{data['detected']}/{data['total']}</td>
            <td>{format_value(data['mean_error'])}</td>
            <td>{format_value(data.get('mean_error_x'))}</td>
            <td>{format_value(data.get('mean_error_y'))}</td>
            <td>{format_value(data.get('mean_error_z'))}</td>
            <td>{format_value(data['median_error'])}</td>
            <td>{format_value(data['max_error'])}</td>
            <td>{format_value(data['inlier_rate_5mm'])}</td>
        </tr>
        """
        landmark_rows.append(row)

    # Generate case rows with resolution information
    case_rows = []
    for case, data in sorted_cases:
        case_errors = [e for e in detailed_errors if e['image'] == case]
        if case_errors:
            spacing = [
                case_errors[0]['resolution_x'],
                case_errors[0]['resolution_y'],
                case_errors[0]['resolution_z']
            ]
            spacing_str = f"{spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f}"
        else:
            spacing_str = 'No detections'
        
        row = f"""
        <tr>
            <td>{case}</td>
            <td>{data['detection_rate']:.2f}%</td>
            <td>{data['detected']}/{data['total']}</td>
            <td>{spacing_str}</td>
        </tr>
        """
        case_rows.append(row)

    # Generate final HTML
    formatted_html = html_content.format(
        model_name=model_name,
        total_landmarks=total_landmarks,
        avg_detection_rate=avg_detection_rate,
        median_detection_rate=median_detection_rate,
        overall_mean_error=overall_mean_error,
        overall_median_error=overall_median_error,
        overall_std_error=overall_std_error,
        inlier_rate_5mm=inlier_rate_5mm,
        landmark_rows="\n".join(landmark_rows),
        case_rows="\n".join(case_rows)
    )

    # Save HTML report
    output_file = os.path.join(output_folder, f'landmark_detection_report_{model_name}.html')
    with open(output_file, 'w') as f:
        f.write(formatted_html)

def generate_csv_reports(landmark_data: Dict[str, Dict[str, float]], 
                        case_data: Dict[str, Dict[str, float]], 
                        detailed_errors: List[Dict],
                        output_folder: str):
    """Generate detailed CSV reports including axis-specific analysis."""
    # Generate landmark summary CSV
    landmark_df = pd.DataFrame([
        {
            'landmark': landmark,
            'detection_rate': data['detection_rate'],
            'detected': data['detected'],
            'total': data['total'],
            'mean_error': data['mean_error'],
            'median_error': data['median_error'],
            'std_error': data['std_error'],
            'max_error': data['max_error'],
            'mean_error_x': data.get('mean_error_x', 'N/A'),
            'mean_error_y': data.get('mean_error_y', 'N/A'),
            'mean_error_z': data.get('mean_error_z', 'N/A'),
            'percentile_25': data['percentile_25'],
            'percentile_75': data['percentile_75'],
            'percentile_90': data['percentile_90'],
            'inlier_rate_5mm': data['inlier_rate_5mm']
        }
        for landmark, data in landmark_data.items()
    ])
    landmark_df.to_csv(os.path.join(output_folder, 'landmark_summary.csv'), index=False)

    # Generate case summary CSV
    case_df = pd.DataFrame([
        {
            'case': case,
            'detection_rate': data['detection_rate'],
            'detected': data['detected'],
            'total': data['total']
        }
        for case, data in case_data.items()
    ])
    case_df.to_csv(os.path.join(output_folder, 'case_summary.csv'), index=False)

    # Generate detailed errors CSV
    detailed_df = pd.DataFrame(detailed_errors)
    detailed_df.to_csv(os.path.join(output_folder, 'detailed_errors.csv'), index=False)

def generate_summary_plots(landmark_data: Dict[str, Dict[str, float]], 
                         case_data: Dict[str, Dict[str, float]], 
                         detailed_errors: List[Dict],
                         output_folder: str):
    """Generate comprehensive visualization plots."""
    plt.style.use('default')

    # 1. Overall error distribution
    plt.figure(figsize=(12, 6))
    errors = [error['error_magnitude_mm'] for error in detailed_errors]
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Landmark Detection Errors')
    plt.xlabel('Error (mm)')
    plt.ylabel('Frequency')
    plt.axvline(np.median(errors), color='r', linestyle='--', label=f'Median: {np.median(errors):.2f}mm')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'error_distribution.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # 2. Axis-specific error distributions
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    x_errors = [abs(error['x_error_mm']) for error in detailed_errors]
    y_errors = [abs(error['y_error_mm']) for error in detailed_errors]
    z_errors = [abs(error['z_error_mm']) for error in detailed_errors]

    ax1.hist(x_errors, bins=30, edgecolor='black', alpha=0.7)
    ax1.set_title('X-Axis Errors')
    ax1.set_xlabel('Error (mm)')
    ax1.set_ylabel('Frequency')

    ax2.hist(y_errors, bins=30, edgecolor='black', alpha=0.7)
    ax2.set_title('Y-Axis Errors')
    ax2.set_xlabel('Error (mm)')

    ax3.hist(z_errors, bins=30, edgecolor='black', alpha=0.7)
    ax3.set_title('Z-Axis Errors')
    ax3.set_xlabel('Error (mm)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'axis_error_distribution.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # 3. Detection rate distribution
    plt.figure(figsize=(12, 6))
    detection_rates = [data['detection_rate'] for data in landmark_data.values()]
    plt.hist(detection_rates, bins=20, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Landmark Detection Rates')
    plt.xlabel('Detection Rate (%)')
    plt.ylabel('Number of Landmarks')
    plt.savefig(os.path.join(output_folder, 'detection_rate_distribution.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # 4. Box plot of errors by axis
    plt.figure(figsize=(10, 6))
    box_data = [x_errors, y_errors, z_errors]
    plt.boxplot(box_data, labels=['X', 'Y', 'Z'])
    plt.title('Error Distribution by Axis')
    plt.ylabel('Error (mm)')
    plt.savefig(os.path.join(output_folder, 'error_boxplot.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # 5. Resolution distribution
    plt.figure(figsize=(15, 5))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    resolutions_x = [error['resolution_x'] for error in detailed_errors]
    resolutions_y = [error['resolution_y'] for error in detailed_errors]
    resolutions_z = [error['resolution_z'] for error in detailed_errors]

    ax1.hist(resolutions_x, bins=20, edgecolor='black', alpha=0.7)
    ax1.set_title('X Resolution Distribution')
    ax1.set_xlabel('Resolution (mm)')
    ax1.set_ylabel('Frequency')
    
    ax2.hist(resolutions_y, bins=20, edgecolor='black', alpha=0.7)
    ax2.set_title('Y Resolution Distribution')
    ax2.set_xlabel('Resolution (mm)')
    
    ax3.hist(resolutions_z, bins=20, edgecolor='black', alpha=0.7)
    ax3.set_title('Z Resolution Distribution')
    ax3.set_xlabel('Resolution (mm)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'resolution_distribution.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # 6. Error vs Resolution scatter plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.scatter([e['resolution_x'] for e in detailed_errors], 
                [abs(e['x_error_mm']) for e in detailed_errors], 
                alpha=0.5)
    ax1.set_title('X Error vs X Resolution')
    ax1.set_xlabel('Resolution (mm)')
    ax1.set_ylabel('Error (mm)')
    
    ax2.scatter([e['resolution_y'] for e in detailed_errors], 
                [abs(e['y_error_mm']) for e in detailed_errors], 
                alpha=0.5)
    ax2.set_title('Y Error vs Y Resolution')
    ax2.set_xlabel('Resolution (mm)')
    
    ax3.scatter([e['resolution_z'] for e in detailed_errors], 
                [abs(e['z_error_mm']) for e in detailed_errors], 
                alpha=0.5)
    ax3.set_title('Z Error vs Z Resolution')
    ax3.set_xlabel('Resolution (mm)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'error_vs_resolution.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # 7. Top 10 landmarks by error
    plt.figure(figsize=(12, 6))
    landmark_mean_errors = {
        landmark: np.mean(data['errors']) 
        for landmark, data in landmark_data.items() 
        if isinstance(data.get('mean_error'), (int, float))
    }
    
    top_10_errors = dict(sorted(landmark_mean_errors.items(), 
                               key=lambda x: x[1], 
                               reverse=True)[:10])
    
    plt.bar(top_10_errors.keys(), top_10_errors.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 10 Landmarks by Mean Error')
    plt.xlabel('Landmark')
    plt.ylabel('Mean Error (mm)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'top_10_errors.png'), bbox_inches='tight', dpi=300)
    plt.close()

def generate_detailed_analysis(landmark_data: Dict[str, Dict[str, float]], 
                             case_data: Dict[str, Dict[str, float]], 
                             detailed_errors: List[Dict],
                             output_folder: str):
    """Generate additional statistical analysis and insights."""
    analysis_results = {
        'overall_statistics': {},
        'axis_specific': {},
        'resolution_impact': {},
        'outliers': {},
        'correlations': {}
    }
    
    # Overall statistics
    all_errors = [error['error_magnitude_mm'] for error in detailed_errors]
    if all_errors:
        analysis_results['overall_statistics'] = {
            'mean_error': np.mean(all_errors),
            'median_error': np.median(all_errors),
            'std_error': np.std(all_errors),
            'min_error': np.min(all_errors),
            'max_error': np.max(all_errors),
            'inlier_rate_5mm': np.mean(np.array(all_errors) < 5) * 100
        }

    # Axis-specific analysis
    for axis in ['x', 'y', 'z']:
        errors = [abs(error[f'{axis}_error_mm']) for error in detailed_errors]
        resolutions = [error[f'resolution_{axis}'] for error in detailed_errors]
        
        analysis_results['axis_specific'][axis] = {
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'std_error': np.std(errors),
            'mean_resolution': np.mean(resolutions),
            'std_resolution': np.std(resolutions)
        }

    # Resolution impact analysis
    for axis in ['x', 'y', 'z']:
        errors = [abs(error[f'{axis}_error_mm']) for error in detailed_errors]
        resolutions = [error[f'resolution_{axis}'] for error in detailed_errors]
        
        if len(errors) > 1 and len(resolutions) > 1:  # Need at least 2 points for correlation
            correlation = np.corrcoef(errors, resolutions)[0, 1]
            analysis_results['resolution_impact'][f'{axis}_axis_correlation'] = correlation

    # Outlier analysis (points beyond 3 standard deviations)
    all_errors = np.array([error['error_magnitude_mm'] for error in detailed_errors])
    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    outlier_threshold = mean_error + 3 * std_error
    
    outliers = [error for error in detailed_errors if error['error_magnitude_mm'] > outlier_threshold]
    analysis_results['outliers'] = {
        'count': len(outliers),
        'percentage': (len(outliers) / len(detailed_errors)) * 100 if detailed_errors else 0,
        'threshold': outlier_threshold,
        'details': outliers
    }

    # Save detailed analysis to JSON
    with open(os.path.join(output_folder, 'detailed_analysis.json'), 'w') as f:
        json.dump(analysis_results, f, indent=2)

    # Generate summary markdown
    # Generate each section separately to avoid f-string with backslash issues
    overall_stats = f"""# Detailed Analysis Report

## Overall Statistics
- Mean Error: {analysis_results['overall_statistics'].get('mean_error', 'N/A'):.2f} mm
- Median Error: {analysis_results['overall_statistics'].get('median_error', 'N/A'):.2f} mm
- Standard Deviation: {analysis_results['overall_statistics'].get('std_error', 'N/A'):.2f} mm
- 5mm Inlier Rate: {analysis_results['overall_statistics'].get('inlier_rate_5mm', 'N/A'):.2f}%\n"""

    axis_specific = "\n## Axis-Specific Analysis\n"
    for axis, stats in analysis_results['axis_specific'].items():
        axis_specific += f"""
### {axis.upper()}-Axis
- Mean Error: {stats['mean_error']:.2f} mm
- Median Error: {stats['median_error']:.2f} mm
- Standard Deviation: {stats['std_error']:.2f} mm
- Mean Resolution: {stats['mean_resolution']:.3f} mm
- Resolution Std: {stats['std_resolution']:.3f} mm\n"""

    resolution_impact = "\n## Resolution Impact\n"
    for axis, corr in analysis_results['resolution_impact'].items():
        resolution_impact += f"- {axis.split('_')[0].upper()}-axis correlation: {corr:.3f}\n"

    outlier_analysis = f"""
## Outlier Analysis
- Number of outliers: {analysis_results['outliers']['count']}
- Percentage of outliers: {analysis_results['outliers']['percentage']:.2f}%
- Outlier threshold: {analysis_results['outliers']['threshold']:.2f} mm
"""

    markdown_content = overall_stats + axis_specific + resolution_impact + outlier_analysis

    with open(os.path.join(output_folder, 'detailed_analysis.md'), 'w') as f:
        f.write(markdown_content)

def generate_reports(landmark_data: Dict[str, Dict[str, float]], 
                    case_data: Dict[str, Dict[str, float]], 
                    detailed_errors: List[Dict],
                    output_folder: str,
                    model_name: str):
    """Generate HTML, CSV reports and visualization plots."""
    # Generate HTML report
    generate_html_report(landmark_data, case_data, detailed_errors, output_folder, model_name)
    
    # Generate CSV reports
    generate_csv_reports(landmark_data, case_data, detailed_errors, output_folder)
    
    # Generate visualization plots
    generate_summary_plots(landmark_data, case_data, detailed_errors, output_folder)

def main():
    args = parse_arguments()
    
    os.makedirs(args.output_folder, exist_ok=True)
    # Extract model name from detection folder
    model_name = extract_model_name(args.detection_folder)
    
    # Load landmark files
    label_files = load_landmark_files(args.label_folder)
    label_landmarks = load_landmarks(label_files)
    
    # Handle detection files if provided
    if args.detection_folder:
        detection_files = load_landmark_files(args.detection_folder)
        common_label_files, common_detection_files = find_common_files(label_files, detection_files)
        label_landmarks = load_landmarks(common_label_files)
        detection_landmarks = load_landmarks(common_detection_files)
        compare_mode = True
    else:
        detection_landmarks = None
        compare_mode = False
    
    # Analyze landmarks
    landmark_data, case_data, detailed_errors = analyze_landmarks(
        label_landmarks, detection_landmarks, args.image_folder, args.debug)
    
    # Generate reports and visualizations
    generate_reports(landmark_data, case_data, detailed_errors, args.output_folder, model_name)
    
    # Generate images if requested
    if args.generate_pictures:
        print('Generating planes for the labelled landmarks.')
        gen_plane_images(args.image_folder, label_landmarks, 'label',
                         args.contrast_range, None, args.output_folder)  # Resolution now comes from image
        
        if compare_mode:
            print('Generating planes for the detected landmarks.')
            gen_plane_images(args.image_folder, detection_landmarks, 'detection',
                             args.contrast_range, None, args.output_folder)  # Resolution now comes from image

if __name__ == '__main__':
    main()

# python scripts/gen_html_report_v4.py --image_folder /home/juval.gutknecht/Projects/Data/Dataset012_aligned/imagesTs --label_folder /home/juval.gutknecht/Projects/Data/Dataset012_aligned/landmarksTs_csv --detection_folder /home/juval.gutknecht/Projects/Data/results/inference_model_comparison/newmask_012_202020  --output_folder /home/juval.gutknecht/Projects/Data/results/inference_model_comparison/newmask_012_202020/report 
