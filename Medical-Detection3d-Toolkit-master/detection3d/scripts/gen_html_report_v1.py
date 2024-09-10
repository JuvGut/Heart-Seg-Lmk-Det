import argparse
import glob
import os
import pandas as pd
import sys
import numpy as np
sys.path.append("..")
sys.path.append(".")
from detection3d.vis.gen_images import gen_plane_images, load_coordinates_from_csv
from detection3d.vis.gen_html_report import gen_html_report


def read_landmark_names(landmark_names_file):
    """
    Read the landmark names from a csv file.
    """
    landmark_names_dict = {}
    df = pd.read_csv(landmark_names_file)
    landmark_idx = list(df['landmark_idx'])
    landmark_name = list(df['landmark_name'])
    for idx in range(len(landmark_idx)):
        landmark_names_dict.update({idx: landmark_name[idx]})

    return landmark_names_dict


def parse_and_check_arguments():
    """
    Parse input arguments and raise error if invalid.
    """
    # default_image_folder = '/home/juval.gutknecht/Projects/Data/A_Subset_012_a/imagesTr'
    # default_label_folder = '/home/juval.gutknecht/Projects/Data/A_Subset_012_a/landmarksTr_csv'
    # default_detection_folder = '/home/juval.gutknecht/Projects/Data/results/inference_results'
    # default_resolution = [1.5, 1.5, 1.5]
    # default_contrast_range = None
    # default_output_folder = '/home/juval.gutknecht/Projects/Data/results/inference_results/html_report'
    # default_generate_pictures = False
    
    default_image_folder = '/home/juval.gutknecht/Projects/Data/Dataset012_USB_Heart_big/imagesTs'
    default_label_folder = '/home/juval.gutknecht/Projects/Data/Dataset012_USB_Heart_big/landmarksTs_csv'
    default_detection_folder = '/home/juval.gutknecht/Projects/Data/results/inference_results_large'
    default_resolution = [0.43, 0.3, 0.43] # [1.5, 1.5, 1.5]
    default_contrast_range = None
    default_output_folder = '/home/juval.gutknecht/Projects/Z_NEW_Medical-Detection3d-Toolkit/detection3d/saves'
    default_generate_pictures = False

    parser = argparse.ArgumentParser(
        description='Snapshot three planes centered around landmarks.')
    parser.add_argument('--image_folder', type=str,
                        default=default_image_folder,
                        help='Folder containing the source data.')
    parser.add_argument('--label_folder', type=str,
                        default=default_label_folder,
                        help='A folder where CSV files containing labelled landmark coordinates are stored.')
    parser.add_argument('--detection_folder', type=str,
                        default=default_detection_folder,
                        help='A folder where CSV files containing detected or baseline landmark coordinates are stored.')
    parser.add_argument('--resolution', type=list,
                        default=default_resolution,
                        help="Resolution of the snap shot images.")
    parser.add_argument('--contrast_range', type=list,
                        default=default_contrast_range,
                        help='Minimal and maximal value of contrast intensity window.')
    parser.add_argument('--output_folder', type=str,
                        default=default_output_folder,
                        help='Folder containing the generated snapshot images.')
    parser.add_argument('--generate_pictures', type=bool,
                        default=default_generate_pictures,
                        help='Folder containing the generated snapshot images.')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_and_check_arguments()
    if not os.path.isdir(args.detection_folder):
        print("The detection folder does not exist, so we only check labelled landmarks.")
        usage_flag = 1
    else:
        print("The detection_folder exists, so we compare the labelled and detected landmarks.")
        usage_flag = 2
    
    print("The label folder: {}".format(args.label_folder))
    label_landmark_csvs = glob.glob(os.path.join(args.label_folder, "*.csv"))
    assert len(label_landmark_csvs) > 0
    label_landmark_csvs.sort()
    print("# landmark files in the label folder: {}".format(len(label_landmark_csvs)))
    
    if usage_flag == 2:
        print("The detection folder: {}".format(args.detection_folder))
        detection_landmark_csvs = glob.glob(os.path.join(args.detection_folder, "*.csv"))
        assert len(detection_landmark_csvs) > 0
        detection_landmark_csvs.sort()
        print("# landmark files in the detection folder: {}".format(
            len(detection_landmark_csvs)))
    
        # find the intersection of the labelled and the detected files
        label_landmark_csvs_folder = os.path.dirname(label_landmark_csvs[0])
        detection_landmark_csvs_folder = os.path.dirname(detection_landmark_csvs[0])

        label_landmark_csvs_basenames = []
        for label_landmark_csv in label_landmark_csvs:
            basename = os.path.basename(label_landmark_csv).split('_')[0]  # Get the BS-XXX part
            label_landmark_csvs_basenames.append(basename)

        detection_landmark_csvs_basenames = []
        for detection_landmark_csv in detection_landmark_csvs:
            basename = os.path.basename(detection_landmark_csv).split('_')[0]  # Get the BS-XXX part
            detection_landmark_csvs_basenames.append(basename)
        
        intersect_basename = \
            list(set(label_landmark_csvs_basenames) & set(detection_landmark_csvs_basenames))
        assert len(intersect_basename) > 0

        label_landmark_csvs, detection_landmark_csvs = [], []
        for basename in intersect_basename:
            label_csv = next(csv for csv in glob.glob(os.path.join(label_landmark_csvs_folder, f"{basename}*.csv")))
            detection_csv = next(csv for csv in glob.glob(os.path.join(detection_landmark_csvs_folder, f"{basename}*.csv")))
            label_landmark_csvs.append(label_csv)
            detection_landmark_csvs.append(detection_csv)
        
        print("# landmark files in both folders: {}".format(
            len(detection_landmark_csvs)))
        
    label_landmarks = {}
    for label_landmark_csv in label_landmark_csvs:
        file_name = os.path.basename(label_landmark_csv).split('_')[0]  # Get the BS-XXX part
        landmarks = load_coordinates_from_csv(label_landmark_csv)
        label_landmarks.update({file_name: landmarks})
    
    if not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)
    
    if usage_flag == 2:
        detection_landmarks = {}
        for detection_landmark_csv in detection_landmark_csvs:
            file_name = os.path.basename(detection_landmark_csv).split('_')[0]  # Get the BS-XXX part
            landmarks = load_coordinates_from_csv(detection_landmark_csv)
            detection_landmarks.update({file_name: landmarks})

        # only consider the images and landmarks in both set
        detection_image_names = set(detection_landmarks.keys())
        label_image_names = set(label_landmarks.keys())
        image_names = detection_image_names & label_image_names

        detection_landmark_names = set(detection_landmarks[list(image_names)[0]].keys())
        label_landmark_names = set(label_landmarks[list(image_names)[0]].keys())
        landmark_names = label_landmark_names & detection_landmark_names

        _detection_landmarks = {}
        for case in image_names:
            __detection_landmarks = {}
            for landmark_name in detection_landmarks[case].keys():
                if landmark_name in landmark_names:
                    __detection_landmarks.update({landmark_name: detection_landmarks[case][landmark_name]})
            _detection_landmarks.update({case: __detection_landmarks})
        detection_landmarks = _detection_landmarks

        _label_landmarks = {}
        for case in image_names:
            __label_landmarks = {}
            for landmark_name in label_landmarks[case].keys():
                if landmark_name in landmark_names:
                    __label_landmarks.update({landmark_name: label_landmarks[case][landmark_name]})
            _label_landmarks.update({case: __label_landmarks})
        label_landmarks = _label_landmarks

        print("# landmarks in each landmark files in both folders: {}".format(
            len(landmark_names)))

    # Generate landmark html report for each landmark.
    landmark_list = [label_landmarks]
    if usage_flag == 2:
        landmark_list.append(detection_landmarks)

    gen_html_report(landmark_list, usage_flag, args.output_folder)
    
    if args.generate_pictures:
        print('Start generating planes for the labelled landmarks.')
        gen_plane_images(args.image_folder, label_landmarks, 'label',
                         args.contrast_range, args.resolution, args.output_folder)
    
        if usage_flag == 2:
            print('Start generating planes for the detected landmarks.')
            gen_plane_images(args.image_folder, detection_landmarks, 'detection',
                             args.contrast_range, args.resolution, args.output_folder)

def gen_html_report(landmark_list, usage_flag, output_folder):
    """
    Generate HTML report for landmark detection results.
    :param landmark_list: A list of dictionaries containing landmark detection results.
    :param usage_flag: An integer flag indicating the usage mode.
    :param output_folder: The folder to save the generated HTML report.
    """
    # Process landmark data
    all_landmarks = set()
    for landmarks in landmark_list:
        all_landmarks.update(landmarks[list(landmarks.keys())[0]].keys())
    
    landmark_data = {}
    for landmark in all_landmarks:
        landmark_data[landmark] = {'detected': 0, 'total': 0, 'mean_error': 0}

    case_data = {}
    for case in landmark_list[0].keys():
        case_data[case] = {'total': 0, 'detected': 0}

    for landmarks in landmark_list:
        for case, case_landmarks in landmarks.items():
            case_data[case]['total'] = len(case_landmarks)
            for landmark, coords in case_landmarks.items():
                landmark_data[landmark]['total'] += 1
                
                # Check if the landmark was detected (not [-1, -1, -1])
                if not np.allclose(coords[:3], [-1, -1, -1]):
                    landmark_data[landmark]['detected'] += 1
                    case_data[case]['detected'] += 1
                    
                    if usage_flag == 2 and len(landmark_list) == 2:
                        error = np.linalg.norm(np.array(landmark_list[0][case][landmark][:3]) - 
                                               np.array(landmark_list[1][case][landmark][:3]))
                        landmark_data[landmark]['mean_error'] += error

    # Calculate mean errors and detection rates
    for landmark in landmark_data:
        if landmark_data[landmark]['detected'] > 0:
            landmark_data[landmark]['mean_error'] /= landmark_data[landmark]['detected']
        landmark_data[landmark]['detection_rate'] = (landmark_data[landmark]['detected'] / 
                                                     landmark_data[landmark]['total']) * 100

    for case in case_data:
        case_data[case]['detection_rate'] = (case_data[case]['detected'] / 
                                             case_data[case]['total']) * 100

    # Sort landmarks and cases by detection rate
    sorted_landmarks = sorted(landmark_data.items(), 
                              key=lambda x: x[1]['detection_rate'], 
                              reverse=True)
    sorted_cases = sorted(case_data.items(), 
                          key=lambda x: x[1]['detection_rate'], 
                          reverse=True)

    # Generate HTML content
    html_content = """
    <html>
    <head>
        <style>
            table, th, td {
                border: 1px solid black;
                border-collapse: collapse;
                padding: 5px;
            }
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
                {0}
            </tr>
            {1}
        </table>

        <h2>Case Detection Summary</h2>
        <table>
            <tr>
                <th>Case</th>
                <th>Detection Rate</th>
                <th>Detected</th>
                <th>Total</th>
            </tr>
            {2}
        </table>
    </body>
    </html>
    """.format(
        "<th>Mean Error</th>" if usage_flag == 2 else "",
        "\n".join([
            "<tr><td>{0}</td><td>{1:.2f}%</td><td>{2}</td><td>{3}</td>{4}</tr>".format(
                landmark, data['detection_rate'], data['detected'], data['total'],
                "<td>{:.2f}</td>".format(data['mean_error']) if usage_flag == 2 else ""
            ) for landmark, data in sorted_landmarks
        ]),
        "\n".join([
            "<tr><td>{0}</td><td>{1:.2f}%</td><td>{2}</td><td>{3}</td></tr>".format(
                case, data['detection_rate'], data['detected'], data['total']
            ) for case, data in sorted_cases
        ])
    )

    # Save the HTML report
    output_file = os.path.join(output_folder, 'landmark_detection_report.html')
    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"HTML report generated and saved to {output_file}")