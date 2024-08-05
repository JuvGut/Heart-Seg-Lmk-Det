import os
import pandas as pd
import random


def split_dataset(image_list, image_folder, output_folder):
    """
    Generate dataset
    """
    seed = 0
    random.Random(seed).shuffle(image_list)

    num_training_images = int(len(image_list) * 5 // 5)
    training_images = image_list[:num_training_images]
    # test_images = image_list[num_training_images:]

    # generate dataset for the training set
    content = []
    training_images.sort()
    print('Generating competition test set ...')
    for name in training_images:
        print(name)
        image_path = os.path.join(image_folder, name, 'img.nrrd')
        content.append([name, image_path])

    csv_file_path = os.path.join(output_folder, 'competition.csv')
    columns = ['image_name', 'image_path']
    df = pd.DataFrame(data=content, columns=columns)
    df.to_csv(csv_file_path, index=False)

    # # generate dataset for the test set
    # content = []
    # test_images.sort()
    # print('Generating test set ...')
    # for name in test_images:
    #     print(name)
    #     image_path = os.path.join(image_folder, name, 'img.nrrd')
    #     landmark_file_path = os.path.join(landmark_file_folder, '{}.csv'.format(name))
    #     landmark_mask_path = os.path.join(landmark_mask_folder, '{}.nii.gz'.format(name))
    #     content.append([name, image_path, landmark_file_path, landmark_mask_path])
    #
    # csv_file_path = os.path.join(output_folder, 'test.csv')
    # columns = ['image_name', 'image_path', 'landmark_file_path', 'landmark_mask_path']
    # df = pd.DataFrame(data=content, columns=columns)
    # df.to_csv(csv_file_path, index=False)


def get_image_list(image_folder):
    """
    Get image list from the image folder
    """
    image_list = []

    images = os.listdir(image_folder)
    for image in images:
        if image.startswith('case'):
            image_list.append(image)

    return image_list


if __name__ == '__main__':

    image_list = get_image_list('/home/mialab22.team2/CSA/DATA/competition/img')
    split_dataset(image_list,
                  '/home/mialab22.team2/CSA/DATA/competition/img',
                  '/home/mialab22.team2/CSA/DATA/competition/test_file')

    # image_list = get_image_list('/home/mialab22.team2/CSA/DATA/img')
    # split_dataset(image_list,
    #               '/home/mialab22.team2/CSA/DATA/img',
    #               '/home/mialab22.team2/CSA/DATA/training_file')

