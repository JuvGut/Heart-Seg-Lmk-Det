import os
import pandas as pd
import random

def split_dataset(image_list, image_folder, landmark_file_folder, landmark_mask_folder, output_folder):
    """
    Generate dataset
    """
    seed = 0
    random.Random(seed).shuffle(image_list)

    num_training_images = int(len(image_list) * 10 // 10)  # Using all images for training in this case
    training_images = image_list[:num_training_images]
    test_images = image_list[num_training_images:]

    # generate dataset for the training set
    content = []
    training_images.sort()
    print('Generating training set ...')
    for name in training_images:
        print(name)
        image_path = os.path.join(image_folder, name)
        landmark_file = find_matching_file(landmark_file_folder, name[:6], '.csv')
        landmark_file_path = os.path.join(landmark_file_folder, landmark_file) if landmark_file else ''
        landmark_mask = find_matching_file(landmark_mask_folder, name[:6], '.nii.gz')
        landmark_mask_path = os.path.join(landmark_mask_folder, landmark_mask) if landmark_mask else ''
        content.append([name, image_path, landmark_file_path, landmark_mask_path])

    csv_file_path = os.path.join(output_folder, 'train.csv')
    columns = ['image_name', 'image_path', 'landmark_file_path', 'landmark_mask_path']
    df = pd.DataFrame(data=content, columns=columns)
    df.to_csv(csv_file_path, index=False)

    # generate dataset for the test set (if any)
    if test_images:
        content = []
        test_images.sort()
        print('Generating test set ...')
        for name in test_images:
            print(name)
            image_path = os.path.join(image_folder, name)
            landmark_file = find_matching_file(landmark_file_folder, name[:6], '.csv')
            landmark_file_path = os.path.join(landmark_file_folder, landmark_file) if landmark_file else ''
            landmark_mask = find_matching_file(landmark_mask_folder, name[:6], '.nii.gz')
            landmark_mask_path = os.path.join(landmark_mask_folder, landmark_mask) if landmark_mask else ''
            content.append([name, image_path, landmark_file_path, landmark_mask_path])

        csv_file_path = os.path.join(output_folder, 'test.csv')
        columns = ['image_name', 'image_path', 'landmark_file_path', 'landmark_mask_path']
        df = pd.DataFrame(data=content, columns=columns)
        df.to_csv(csv_file_path, index=False)

def find_matching_file(folder, prefix, extension):
    """
    Find a file in the given folder that starts with the prefix and has the given extension
    """
    for filename in os.listdir(folder):
        if filename.startswith(prefix) and filename.endswith(extension):
            return filename
    return None

def get_image_list(image_folder):
    """
    Get image list from the image folder
    """
    image_list = []

    for filename in os.listdir(image_folder):
        if filename.endswith('.nrrd'):
            image_list.append(filename)

    return image_list

if __name__ == '__main__':
    image_folder = '/home/juval.gutknecht/Projects/Storage_server/Heart-Project/nnUNet_raw/Dataset012_USB_Heart_big/imagesTr'
    landmark_file_folder = '/home/juval.gutknecht/Projects/Storage_server/Heart-Project/nnUNet_raw/Dataset012_USB_Heart_big/landmarksTr_csv'
    landmark_mask_folder = '/home/juval.gutknecht/Projects/Storage_server/Heart-Project/nnUNet_raw/Dataset012_USB_Heart_big/landmark/mask'
    output_folder = '/home/juval.gutknecht/Projects/Storage_server/Heart-Project/nnUNet_raw/Dataset012_USB_Heart_big/training_file'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_list = get_image_list(image_folder)
    split_dataset(image_list, image_folder, landmark_file_folder, landmark_mask_folder, output_folder)

    print("Dataset generation complete. Check the output folder for train.csv and test.csv (if applicable).")