import SimpleITK as sitk
import os

data_path = '/home/mialab22.team2/CSA/DATA/competition/datasets'
img_output = '/home/mialab22.team2/CSA/DATA/competition/img'

datasets = os.listdir(data_path)
number = 1
for dataset in datasets:
    files = os.listdir(os.path.join(data_path, dataset))
    for file in files:
        path, file_format = os.path.splitext(file)
        if file_format == '.nrrd':
            img = sitk.ReadImage(os.path.join(data_path, dataset, file))
            print(os.path.join(data_path, dataset, file))
            save_path = os.path.join(img_output, 'case_competition_' + str(number))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            sitk.WriteImage(img, os.path.join(save_path, 'img.nrrd'))
    number += 1
