import torchio as tio
import SimpleITK as sitk
import os
import pandas as pd

img_path = '/home/mialab22.team2/CSA/DATA/img'
mask_path = '/home/mialab22.team2/CSA/DATA/landmark/mask'
img_output = '/home/mialab22.team2/CSA/DATA/img'
mask_output = '/home/mialab22.team2/CSA/DATA/landmark/mask'
coords = '/home/mialab22.team2/CSA/DATA/landmark/coords'
coords_output = '/home/mialab22.team2/CSA/DATA/landmark/coords'

img_list = os.listdir(img_path)
mask_list = os.listdir(mask_path)


flip_L = tio.Flip(axes='L',)
flip_A = tio.Flip(axes='A',)
noise = tio.Noise(mean=50, std=20, seed=10)
translation = tio.Affine(translation=[10, 10 , 10], center='image', scales=1, degrees=0)
rotation = tio.Affine(degrees=[10, 10 , 10], center='image', scales=1, translation=0)


# Left Flip
for image in img_list:
    img = sitk.ReadImage(os.path.join(img_path, image, 'img.nrrd'))
    print(image)
    flippedimage = flip_L(img)
    os.makedirs(os.path.join(img_output, image + '_flipped_L'))
    sitk.WriteImage(flippedimage, os.path.join(img_output, image + '_flipped_L', 'img.nrrd'))

for image in img_list:
    mask = sitk.ReadImage(os.path.join(mask_path, image + '.nii.gz'))
    flippedmask = flip_L(mask)
    sitk.WriteImage(flippedmask, os.path.join(mask_output, image + '_flipped_L.nii.gz'))

for image in img_list:
    landmark = pd.read_csv(os.path.join(coords, image + '.csv'))
    pd.DataFrame = landmark
    pd.DataFrame.to_csv(os.path.join(coords_output, image + '_flipped_L.csv'))


#Anterior Flip
for image in img_list:
    img = sitk.ReadImage(os.path.join(img_path, image, 'img.nrrd'))
    print(image)
    flippedimage = flip_A(img)
    os.makedirs(os.path.join(img_output, image + '_flipped_A'))
    sitk.WriteImage(flippedimage, os.path.join(img_output, image + '_flipped_A', 'img.nrrd'))

for image in img_list:
    mask = sitk.ReadImage(os.path.join(mask_path, image + '.nii.gz'))
    flippedmask = flip_A(mask)
    sitk.WriteImage(flippedmask, os.path.join(mask_output, image + '_flipped_A.nii.gz'))

for image in img_list:
    landmark = pd.read_csv(os.path.join(coords, image + '.csv'))
    pd.DataFrame = landmark
    pd.DataFrame.to_csv(os.path.join(coords_output, image + '_flipped_A.csv'))


#adding noise
for image in img_list:
    img = sitk.ReadImage(os.path.join(img_path, image, 'img.nrrd'))
    print(image)
    noiseimage = noise(img)
    os.makedirs(os.path.join(img_output, image + '_noise'))
    sitk.WriteImage(noiseimage, os.path.join(img_output, image + '_noise', 'img.nrrd'))

for image in img_list:
    mask = sitk.ReadImage(os.path.join(mask_path, image + '.nii.gz'))
    sitk.WriteImage(mask, os.path.join(mask_output, image + '_noise.nii.gz'))

for image in img_list:
    landmark = pd.read_csv(os.path.join(coords, image + '.csv'))
    pd.DataFrame = landmark
    pd.DataFrame.to_csv(os.path.join(coords_output, image + '_noise.csv'))


#translation
for image in img_list:
    img = sitk.ReadImage(os.path.join(img_path, image, 'img.nrrd'))
    print(image)
    transimage = translation(img)
    os.makedirs(os.path.join(img_output, image + '_translated'))
    sitk.WriteImage(transimage, os.path.join(img_output, image + '_translated', 'img.nrrd'))

for image in img_list:
    mask = sitk.ReadImage(os.path.join(mask_path, image + '.nii.gz'))
    transmask = translation(mask)
    sitk.WriteImage(transmask, os.path.join(mask_output, image + '_translated.nii.gz'))

for image in img_list:
    landmark = pd.read_csv(os.path.join(coords, image + '.csv'))
    pd.DataFrame = landmark
    pd.DataFrame.to_csv(os.path.join(coords_output, image + '_translated.csv'))


#rotation
for image in img_list:
    img = sitk.ReadImage(os.path.join(img_path, image, 'img.nrrd'))
    print(image)
    rotimage = rotation(img)
    os.makedirs(os.path.join(img_output, image + '_rotated'))
    sitk.WriteImage(rotimage, os.path.join(img_output, image + '_rotated', 'img.nrrd'))

for image in img_list:
    mask = sitk.ReadImage(os.path.join(mask_path, image + '.nii.gz'))
    rotmask = rotation(mask)
    sitk.WriteImage(rotmask, os.path.join(mask_output, image + '_rotated.nii.gz'))

for image in img_list:
    landmark = pd.read_csv(os.path.join(coords, image + '.csv'))
    pd.DataFrame = landmark
    pd.DataFrame.to_csv(os.path.join(coords_output, image + '_rotated.csv'))







# #########################
# image = sitk.ReadImage('TEST/input/img.nrrd')
# mask = sitk.ReadImage('TEST/input/case_01.nii.gz')
# print(mask)
# #
# flip = tio.Flip(axes='L',)
# flippedimage=flip(image)
# flippedmask=flip(mask)
#
#
# sitk.WriteImage(flippedimage, 'TEST/output/img_flipped.nrrd')
# sitk.WriteImage(flippedmask, 'TEST/output/case_01_flipped.nii.gz')
# ############################