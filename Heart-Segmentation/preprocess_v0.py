import os
from glob import glob
import shutil
# from tqdm import tqdm
#import dicom2nifti
#import numpy as np
#import nibabel as nib
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Resized,
    EnsureTyped,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
)
from monai.data import DataLoader, Dataset, CacheDataset, SmartCacheDataset
from monai.utils import set_determinism


# ADAPT THE PARAMETERS                             vvv         vvv               vvv   vvv  vvv (1/4th of original) was 128, 128, 122
def prepare(in_dir, pixdim=(5,5,5), a_min=-200, a_max=1000, spatial_size=[128, 128, 128], cache=True):
    set_determinism(seed=0)

    path_train_volumes = sorted(glob(os.path.join(in_dir, "TrainVolumes", "*.nrrd")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "TrainSegmentations", "*.nrrd")))

    path_test_volumes = sorted(glob(os.path.join(in_dir, "TestVolumes", "*.nrrd")))
    path_test_segmentation = sorted(glob(os.path.join(in_dir, "TestSegmentations", "*.nrrd")))

    # Check if the number of volumes and segmentations match
    print(f"Number of train volumes: {len(path_train_volumes)}")
    print(f"Number of train segmentations: {len(path_train_segmentation)}")
    print(f"Number of test volumes: {len(path_test_volumes)}")
    print(f"Number of test segmentations: {len(path_test_segmentation)}")

    train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                   zip(path_train_volumes, path_train_segmentation)]
    test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                  zip(path_test_volumes, path_test_segmentation)]

    transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"], reader="NrrdReader"),
            EnsureChannelFirstd(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["vol", "seg"], source_key="vol"),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),
            EnsureTyped(keys=["vol", "seg"]),
        ]
    )

    if cache:
        print("Preparing cached dataset...")
        train_ds = CacheDataset(data=train_files, transform=transforms, cache_rate=1.0, num_workers=4)
        # train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
        test_ds = CacheDataset(data=test_files, transform=transforms, cache_rate=1.0, num_workers=4)
        # test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=True)
    else:
        print("Preparing non-cached dataset...")
        train_ds = Dataset(data=train_files, transform=transforms)
        # train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
        test_ds = Dataset(data=test_files, transform=transforms)
        # test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=True)

    print("Creating data loaders...")
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=True)

    print(f"Number of training samples: {len(train_ds)}")
    print(f"Number of test samples: {len(test_ds)}")

    return train_loader, test_loader



if __name__ == '__main__':
    data_dir = '/home/juval.gutknecht/Projects/Data/USB'
    result = prepare(data_dir, cache=True)
    print("After calling prepare function:")
    print(f"Result type: {type(result)}")
    if isinstance(result, tuple) and len(result) == 2:
        train_loader, test_loader = result
        print(f"train_loader type: {type(train_loader)}")
        print(f"test_loader type: {type(test_loader)}")
    else:
        print("prepare function did not return expected tuple")
    print('Data prepared')