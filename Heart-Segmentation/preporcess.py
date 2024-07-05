import os
from glob import glob
import shutil
#from tqdm import tqdm
#import dicom2nifti
#import numpy as np
#import nibabel as nib
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,

)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism
from utilities import show_patient

# ADAPT THE PARAMETERS                             vvv         vvv               vvv   vvv  vvv (1/4th of original) was 128, 128, 122
def prepare(in_dir, pixdim=(5,5,5), a_min=-200, a_max=1000, spatial_size=[128, 128, 128], cache=True):
    """
    This function is for preprocessing, it contains only the basic transforms, but you can add more operations that you
    find in the Monai documentation.
    https://monai.io/docs.html
    """

    set_determinism(seed=0)

    path_train_volumes = sorted(glob(os.path.join(in_dir, "TrainVolumes", "*.nrrd")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "TrainSegmentations", "*.nrrd")))

    path_test_volumes = sorted(glob(os.path.join(in_dir, "TestVolumes", "*.nrrd")))
    path_test_segmentation = sorted(glob(os.path.join(in_dir, "TestSegmentations", "*.nrrd")))

    train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                   zip(path_train_volumes, path_train_segmentation)]
    test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                  zip(path_test_volumes, path_test_segmentation)]

    train_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"], reader="NrrdReader"),
            EnsureChannelFirstd(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["vol", "seg"], source_key="vol"),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),
            ToTensord(keys=["vol", "seg"]),

        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"], reader="NrrdReader"),
            EnsureChannelFirstd(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),
            ToTensord(keys=["vol", "seg"]),

        ]
    )

    if cache:
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)

        test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0)
        test_loader = DataLoader(test_ds, batch_size=1, num_workers=8, pin_memory=True)
        
        return train_loader, test_loader

    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)

        test_ds = Dataset(data=test_files, transform=test_transforms)
        test_loader = DataLoader(test_ds, batch_size=1, num_workers=8, pin_memory=True)

        return train_loader, test_loader



if __name__ == '__main__':
    data_dir = '/home/juval.gutknecht/Projects/Data/USB'
    prepare(data_dir, cache=True)
    print('Data prepared')
    #show_patient(data_in)
