import os
import json 
import shutil
import tempfile
import time
from glob import glob

# from tqdm import tqdm
#import dicom2nifti
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import numpy as np
import nibabel as nib

import torch
from monai.config import print_config
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms.compose import MapTransform
from monai.transforms.utils import generate_spatial_bounding_box

from monai.transforms import (
    Activations,
    AsDiscrete,
    CastToTyped,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensity,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandZoomd,
    SpatialCrop,
    SpatialPadd,
    ToTensord,
    EnsureTyped,
)
from monai.utils.enums import MetricReduction
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR, DynUNet
from monai import data
from monai.data import DataLoader, Dataset, CacheDataset, SmartCacheDataset, decollate_batch
from monai.utils import set_determinism
from functools import partial
import SimpleITK as sitk
from skimage.transform import resize
from utilities_v1 import RemapLabels

# print_config()
start_time = time.time()

# Setup data directory
'''
You can specify a directory with the MONAI_DATA_DIRECTORY environment variable.
This allows you to save results and reuse downloads.
If not specified a temporary directory will be used.
'''
directory = os.environ.get("MONAI_DATA_DIRECTORY")
if directory is not None:
    os.makedirs(directory, exist_ok=True)
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)
    
    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]
    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)
    return tr, val

def save_checkpoint(model, epoch, filename="model.pt", best_acc=0, dir_add=root_dir):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)

def resample_image(image, shape, anisotrophy_flag):
    resized_channels = []
    if anisotrophy_flag:
        for image_c in image:
            resized_slices = []
            for i in range(image_c.shape[-1]):
                image_c_2d_slice = image_c[:, :, i]
                image_c_2d_slice = resize(
                    image_c_2d_slice,
                    shape[:-1],
                    order=3,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                resized_slices.append(image_c_2d_slice)
            resized = np.stack(resized_slices, axis=-1)
            resized = resize(
                resized,
                shape,
                order=0,
                mode="constant",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            resized_channels.append(resized)
    else:
        for image_c in image:
            resized = resize(
                image_c,
                shape,
                order=3,
                mode="edge",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            resized_channels.append(resized)
    resized = np.stack(resized_channels, axis=0)
    return resized


def resample_label(label, shape, anisotrophy_flag):
    reshaped = np.zeros(shape, dtype=np.uint8)
    n_class = np.max(label)
    if anisotrophy_flag:
        shape_2d = shape[:-1]
        depth = label.shape[-1]
        reshaped_2d = np.zeros((*shape_2d, depth), dtype=np.uint8)

        for class_ in range(1, int(n_class) + 1):
            for depth_ in range(depth):
                mask = label[0, :, :, depth_] == class_
                resized_2d = resize(
                    mask.astype(float),
                    shape_2d,
                    order=1,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                reshaped_2d[:, :, depth_][resized_2d >= 0.5] = class_
        for class_ in range(1, int(n_class) + 1):
            mask = reshaped_2d == class_
            resized = resize(
                mask.astype(float),
                shape,
                order=0,
                mode="constant",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            reshaped[resized >= 0.5] = class_
    else:
        for class_ in range(1, int(n_class) + 1):
            mask = label[0] == class_
            resized = resize(
                mask.astype(float),
                shape,
                order=1,
                mode="edge",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            reshaped[resized >= 0.5] = class_

    reshaped = np.expand_dims(reshaped, 0)
    return reshaped


def recovery_prediction(prediction, shape, anisotrophy_flag):
    reshaped = np.zeros(shape, dtype=np.uint8)
    n_class = shape[0]
    if anisotrophy_flag:
        c, h, w = prediction.shape[:-1]
        d = shape[-1]
        reshaped_d = np.zeros((c, h, w, d), dtype=np.uint8)
        for class_ in range(1, n_class):
            mask = prediction[class_] == 1
            resized_d = resize(
                mask.astype(float),
                (h, w, d),
                order=0,
                mode="constant",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            reshaped_d[class_][resized_d >= 0.5] = 1

        for class_ in range(1, n_class):
            for depth_ in range(d):
                mask = reshaped_d[class_, :, :, depth_] == 1
                resized_hw = resize(
                    mask.astype(float),
                    shape[1:-1],
                    order=1,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                reshaped[class_, :, :, depth_][resized_hw >= 0.5] = 1
    else:
        for class_ in range(1, n_class):
            mask = prediction[class_] == 1
            resized = resize(
                mask.astype(float),
                shape[1:],
                order=1,
                mode="edge",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            reshaped[class_][resized >= 0.5] = 1

    return reshaped


class PreprocessAnisotropic(MapTransform):
    """
    This transform class takes NNUNet's preprocessing method for reference.
    That code is in:
    https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py

    """

    def __init__(
        self,
        keys,
        clip_values,
        pixdim,
        normalize_values,
        model_mode,
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.low = clip_values[0]
        self.high = clip_values[1]
        self.target_spacing = pixdim
        self.mean = normalize_values[0]
        self.std = normalize_values[1]
        self.training = False
        self.crop_foreg = CropForegroundd(keys=["image", "label"], source_key="image")
        self.normalize_intensity = NormalizeIntensity(nonzero=True, channel_wise=True)
        if model_mode in ["train"]:
            self.training = True

    def calculate_new_shape(self, spacing, shape):
        spacing_ratio = np.array(spacing) / np.array(self.target_spacing)
        new_shape = (spacing_ratio * np.array(shape)).astype(int).tolist()
        return new_shape

    def check_anisotrophy(self, spacing):
        def check(spacing):
            return np.max(spacing) / np.min(spacing) >= 3

        return check(spacing) or check(self.target_spacing)

    def __call__(self, data):
        # load data
        d = dict(data)
        image = d["image"]

        # image_spacings = d["image_meta_dict"]["pixdim"][1:4].tolist()
        # Safely access image_meta_dict
        image_meta_dict = d.get("image_meta_dict", {})
        image_spacings = image_meta_dict.get("pixdim", [0, 1, 1, 1])[1:4].tolist() # <- issues here

        if not any(image_spacings):
            image_spacings = self.target_spacing

        if "label" in self.keys:
            label = d["label"]
            label[label < 0] = 0

        if self.training:
            # only task 04 does not be impacted
            cropped_data = self.crop_foreg({"image": image, "label": label})
            image, label = cropped_data["image"], cropped_data["label"]
        else:
            d["original_shape"] = np.array(image.shape[1:])
            box_start, box_end = generate_spatial_bounding_box(image)
            image = SpatialCrop(roi_start=box_start, roi_end=box_end)(image)
            d["bbox"] = np.vstack([box_start, box_end])
            d["crop_shape"] = np.array(image.shape[1:])

        original_shape = image.shape[1:]
        # calculate shape
        resample_flag = False
        anisotrophy_flag = False

        image = image.numpy()
        if self.target_spacing != image_spacings:
            # resample
            resample_flag = True
            resample_shape = self.calculate_new_shape(image_spacings, original_shape)
            anisotrophy_flag = self.check_anisotrophy(image_spacings)
            image = resample_image(image, resample_shape, anisotrophy_flag)
            if self.training:
                label = resample_label(label, resample_shape, anisotrophy_flag)

        d["resample_flag"] = resample_flag
        d["anisotrophy_flag"] = anisotrophy_flag
        # clip image for CT dataset
        if self.low != 0 or self.high != 0:
            image = np.clip(image, self.low, self.high)
            image = (image - self.mean) / self.std
        else:
            image = self.normalize_intensity(image.copy())

        d["image"] = image

        if "label" in self.keys:
            d["label"] = label

        return d
##### SETUP DATALOADER #####

def get_loader(batch_size, data_dir, json_list, fold, roi):
    data_dir = data_dir
    datalist_json = json_list
    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=fold)
    
    # for d in train_files: ### Check how many unique labels are in the label files ###
    #     label = sitk.GetArrayFromImage(sitk.ReadImage(d["label"]))
    #     print(f"unique values in label: {np.unique(label)}")
    
    load_transforms = [
        LoadImaged(keys=["image", "label"], reader="NrrdReader"),
        EnsureChannelFirstd(keys=["image", "label"]),
        # AddMetaDictd(keys=["image"]),
    ]

    sample_transforms = [
        PreprocessAnisotropic(
            keys=["image", "label"], # image and label share the same key names
            clip_values=[0,0], # "02": [0, 0],
            pixdim=[1.0, 1.0, 1.0], # "02": [1.25, 1.25, 1.37],
            normalize_values=[0, 0], # "02": [0, 0],
            model_mode="train",
        ),
        ToTensord(keys="image"),
    ]
    sample_val_transforms = [
        PreprocessAnisotropic(
            keys=["image", "label"], # image and label share the same key names
            clip_values=[0,0], # "02": [0, 0],
            pixdim=[1.0, 1.0, 1.0], # "02": [1.25, 1.25, 1.37],
            normalize_values=[0, 0], # "02": [0, 0],
            model_mode="val",
        ),
        ToTensord(keys="image"),
    ]
    training_transforms = [
            SpatialPadd(keys=["image", "label"], spatial_size=[128, 128, 128]), # "02": [160, 192, 80],
            # RandCropByPosNegLabeld(
            #     keys=["image", "label"],
            #     label_key="label",
            #     spatial_size=patch_size[task_id],
            #     pos=pos_sample_num,
            #     neg=neg_sample_num,
            #     num_samples=num_samples,
            #     image_key="image",
            #     image_threshold=0,
            # ),
            RandZoomd(
                keys=["image", "label"],
                min_zoom=0.9,
                max_zoom=1.2,
                mode=("trilinear", "nearest"),
                align_corners=(True, None),
                prob=0.15,
            ),
            RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.15),
                sigma_y=(0.5, 1.15),
                sigma_z=(0.5, 1.15),
                prob=0.15,
            ),
            RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15),
            RandFlipd(["image", "label"], spatial_axis=[0], prob=0.5),
            RandFlipd(["image", "label"], spatial_axis=[1], prob=0.5),
            RandFlipd(["image", "label"], spatial_axis=[2], prob=0.5),
            CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8)),
            EnsureTyped(keys=["image", "label"]),
        ]
    validation_transforms = [
            CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8)),
            EnsureTyped(keys=["image", "label"]),
        ]

    train_transform = Compose(load_transforms + 
                              sample_transforms + 
                              training_transforms)
    val_transform = Compose(load_transforms +
                            sample_val_transforms +
                            validation_transforms)


    train_ds = data.Dataset(data=train_files, transform=train_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    val_ds = data.Dataset(data=validation_files, transform=val_transform)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    return train_loader, val_loader

# Set dataset root directory and hyper-parameters
'''The following hyper-parameters are set for the purpose of this tutorial. However, additional changes, as described below, maybe beneficial.

If GPU memory is not sufficient, reduce sw_batch_size to 2 or batch_size to 1.

Decrease val_every (validation frequency) to 1 for obtaining more accurate checkpoints.'''

data_dir = "/home/juval.gutknecht/Projects/Data/USB"
json_list = "/home/juval.gutknecht/Projects/Data/USB/USB.json"
roi = (128, 128, 128)
batch_size = 1
sw_batch_size = 4
fold = 1
infer_overlap = 0.5
max_epochs = 300
val_every = 10
train_loader, val_loader = get_loader(batch_size, data_dir, json_list, fold, roi)

print(f"Number of training samples: {len(train_loader.dataset)}")
print(f"Number of validation samples: {len(val_loader.dataset)}")

# ##### Check data shape and visualize ##### -> Time consuming
# datavisualizetimer = time.time()
# img_path = os.path.join(data_dir, "TrainVolumes", "BS-015.nrrd")
# label_path = os.path.join(data_dir, "TrainSegmentations", "BS-015.seg.nrrd")
# img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
# label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))

# slice_idx = 250
# cmap = mpl.colormaps["viridis"].resampled(5) # 5 discrete colors
# print(f"image shape: {img.shape}")
# print(f"label shape: {label.shape}")
# n_classes = 5
# colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
# colors = np.roll(colors, shift=1, axis=0)
# colors[-1] = [0, 0, 0, 0]  # Make background transparent
# cmap = ListedColormap(colors)
# # Label names (modify these according to your specific labels)
# label_names = ['LVOT', 'RCC', 'LCC', 'NCC', 'Background']
# plt.figure(figsize=(12, 6))
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.set_title("Image with Overlaid Labels")
# ax.imshow(img[slice_idx], cmap="gray")
# ax.imshow(label[slice_idx], cmap=cmap, alpha=0.5, vmin=0, vmax=n_classes-1)
# ax.axis('off')
# patches = [mpl.patches.Patch(color=colors[i], label=label_names[i]) for i in range(n_classes)]
# plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# plt.tight_layout()
# plt.savefig("sample_slice.png", bbox_inches='tight')
# print(f"Image with overlaid labels and legend saved as sample_slice.png and took {(time.time() - datavisualizetimer):.2f} seconds to run.")
# plt.close(fig)

##### Create Swin UNETR model #####
'''In this scetion, we create Swin UNETR model for the 3-class brain tumor semantic segmentation. 
We use a feature size of 48. We also use gradient checkpointing (use_checkpoint) for more memory-
efficient training. However, use_checkpoint for faster training if enough GPU memory is available.'''

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# model = SwinUNETR(
#     img_size=roi,
#     in_channels=4, # 1 or 4 channels
#     out_channels=5, # 5 classes [0 - 4]
#     feature_size=48,
#     drop_rate=0.0,
#     attn_drop_rate=0.0,
#     dropout_path_rate=0.0,
#     use_checkpoint=True,
# ).to(device)

model = DynUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=5, # 5 classes [0 - 4]
    kernel_size=(3, 3, 3),
    strides=(1, 2, 2),
    upsample_kernel_size=(2, 2, 2),
    filters=(32, 64, 128, 256, 512),
).to(device)

##### Set loss function and optimizer #####
torch.backends.cudnn.benchmark = True
dice_loss = DiceLoss(to_onehot_y=True, sigmoid=True)
post_sigmoid = Activations(sigmoid=True)
# post_pred = AsDiscrete(argmax=False, threshold=0.5)
# dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
dice_acc = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
post_pred = AsDiscrete(argmax=True, to_onehot=5)
model_inferer = partial(
    sliding_window_inference,
    roi_size=[roi[0], roi[1], roi[2]],
    sw_batch_size=sw_batch_size,
    predictor=model,
    overlap=infer_overlap,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)



##### Define Train and Validation Epoch #####

def train_epoch(model, loader, optimizer, epoch, loss_func):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(device), batch_data["label"].to(device)
        logits = model(data)

        # print(f"logits shape: {logits.shape}")
        # print(f"target shape: {target.shape}")
        # print(f"unique values in target: {torch.unique(target)}")

        loss = loss_func(logits, target) 
        loss.backward()  ########### the error is here
        optimizer.step()
        run_loss.update(loss.item(), n=batch_size)
        print(
            "Epoch {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - start_time),
        )
        start_time = time.time()
    return run_loss.avg


# def val_epoch(
#     model,
#     loader,
#     epoch,
#     acc_func,
#     model_inferer=None,
#     post_sigmoid=None,
#     post_pred=None,
# ):
#     model.eval()
#     start_time = time.time()
#     run_acc = AverageMeter()

#     with torch.no_grad():
#         for idx, batch_data in enumerate(loader):
#             data, target = batch_data["image"].to(device), batch_data["label"].to(device)
#             logits = model_inferer(data)
#             val_labels_list = decollate_batch(target)
#             val_outputs_list = decollate_batch(logits)
#             val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
#             acc_func.reset()
#             acc_func(y_pred=val_output_convert, y=val_labels_list)
#             acc, not_nans = acc_func.aggregate()
#             run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
#             dice_tc = run_acc.avg[0]
#             dice_wt = run_acc.avg[1]
#             dice_et = run_acc.avg[2]
#             print(
#                 "Val {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
#                 ", dice_tc:",
#                 dice_tc,
#                 ", dice_wt:",
#                 dice_wt,
#                 ", dice_et:",
#                 dice_et,
#                 ", time {:.2f}s".format(time.time() - start_time),
#             )
#             start_time = time.time()

#     return run_acc.avg

def val_epoch(model, loader, epoch, acc_func, model_inferer=None, post_sigmoid=None, post_pred=None):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            logits = model_inferer(data)
            val_outputs = [post_pred(logits)]
            acc_func(y_pred=val_outputs, y=target)
            acc = acc_func.aggregate().item()
            print(
                "Val {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
                ", dice:",
                acc,
                ", time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()
    mean_dice = acc_func.aggregate().item()
    return mean_dice

##### Trainer #####

def trainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    scheduler,
    model_inferer=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
):
    val_acc_max = 0.0
    dices_tc = []
    dices_wt = []
    dices_et = []
    dices_avg = []
    loss_epochs = []
    trains_epoch = []
    for epoch in range(start_epoch, max_epochs):
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            epoch=epoch,
            loss_func=loss_func,
        )
        print(
            "Final training  {}/{}".format(epoch, max_epochs - 1),
            "loss: {:.4f}".format(train_loss),
            "time {:.2f}s".format(time.time() - epoch_time),
        )

        if (epoch + 1) % val_every == 0 or epoch == 0:
            loss_epochs.append(train_loss)
            trains_epoch.append(int(epoch))
            epoch_time = time.time()
            acc_func.reset()
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )
            if isinstance(val_acc, (list, np.ndarray)):
                dice_tc = val_acc[0]
                dice_wt = val_acc[1]
                dice_et = val_acc[2]
                val_avg_acc = np.mean(val_acc)
            else:
                # If it's a single number, use it as the average
                val_avg_acc = val_acc
                dice_tc = dice_wt = dice_et = val_acc  # or set these to None if not applicable
            print(
                "Final validation stats {}/{}".format(epoch, max_epochs - 1),
                ", dice_tc:",
                dice_tc,
                ", dice_wt:",
                dice_wt,
                ", dice_et:",
                dice_et,
                ", Dice_Avg:",
                val_avg_acc,
                ", time {:.2f}s".format(time.time() - epoch_time),
            )
            dices_tc.append(dice_tc)
            dices_wt.append(dice_wt)
            dices_et.append(dice_et)
            dices_avg.append(val_avg_acc)
            if val_avg_acc > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                save_checkpoint(
                    model,
                    epoch,
                    best_acc=val_acc_max,
                )
            scheduler.step()
    print("Training Finished !, Best Accuracy: ", val_acc_max)
    return (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    )

print(f"Until \"Trainer\" it took {(time.time() - start_time):.2f} seconds")

##### Execute training #####

start_epoch = 0

(
    val_acc_max,
    dices_tc,
    dices_wt,
    dices_et,
    dices_avg,
    loss_epochs,
    trains_epoch,
) = trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_func=dice_loss,
    acc_func=dice_acc,
    scheduler=scheduler,
    model_inferer=model_inferer,
    start_epoch=start_epoch,
    post_sigmoid=post_sigmoid,
    post_pred=post_pred,
)


print(f"train completed, best average dice: {val_acc_max:.4f} ")


print(f"To Train it took {(time.time() - start_time):.2f} seconds")


# # ADAPT THE PARAMETERS                             vvv         vvv               vvv   vvv  vvv (1/4th of original) was 128, 128, 122
# def prepare(in_dir, pixdim=(5,5,5), a_min=-200, a_max=1000, spatial_size=[128, 128, 128], cache=True):
#     set_determinism(seed=0)

#     path_train_volumes = sorted(glob(os.path.join(in_dir, "TrainVolumes", "*.nrrd")))
#     path_train_segmentation = sorted(glob(os.path.join(in_dir, "TrainSegmentations", "*.nrrd")))

#     path_test_volumes = sorted(glob(os.path.join(in_dir, "TestVolumes", "*.nrrd")))
#     path_test_segmentation = sorted(glob(os.path.join(in_dir, "TestSegmentations", "*.nrrd")))

#     # Check if the number of volumes and segmentations match
#     print(f"Number of train volumes: {len(path_train_volumes)}")
#     print(f"Number of train segmentations: {len(path_train_segmentation)}")
#     print(f"Number of test volumes: {len(path_test_volumes)}")
#     print(f"Number of test segmentations: {len(path_test_segmentation)}")

#     train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
#                    zip(path_train_volumes, path_train_segmentation)]
#     test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
#                   zip(path_test_volumes, path_test_segmentation)]

#     mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

#     common_transforms = [
#         LoadImaged(keys=["vol", "seg"], reader="NrrdReader"),
#         EnsureChannelFirstd(keys=["vol", "seg"]),
#         ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
#         RemapLabels(keys=["seg"], mapping=mapping),
#                         ]

#     train_transforms = Compose(
#         common_transforms + [
            
#             Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
#             Orientationd(keys=["vol", "seg"], axcodes="RAS"),
#             ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
#             CropForegroundd(keys=["vol", "seg"], source_key="vol"),
#             Resized(keys=["vol", "seg"], spatial_size=spatial_size),
#             EnsureTyped(keys=["vol", "seg"]),
#         ]
#     )

#     test_transforms = Compose(
#         common_transforms + [
#             Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
#             Orientationd(keys=["vol", "seg"], axcodes="RAS"),
#             ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
#             CropForegroundd(keys=["vol", "seg"], source_key="vol"),
#             Resized(keys=["vol", "seg"], spatial_size=spatial_size),
#             RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]), 
#             RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5), 
#             RandZoomd(keys=["image", "label"], min_zoom=0.9, max_zoom=1.1, prob=0.5), 
#             RandAffined(keys=["image", "label"], prob=0.5, rotate_range=(0, 0.3), scale_range=(0.8, 1.2), mode='bilinear'), 
#             SpatialPadd(keys=["image", "label"], spatial_size=[160, 160, 160], mode='constant'), 
#             CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 128]),
#             EnsureTyped(keys=["vol", "seg"]),
#         ]
#     )


#     if cache:
#         print("Preparing smart cached dataset...")
#         train_ds = SmartCacheDataset(data=train_files, transform=train_transforms, cache_num=64, cache_rate=0.1, replace_rate=0.2)
#         test_ds = SmartCacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0)
#     # elif cache:
#     #     print("Preparing cached dataset...")
#     #     train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
#     #     # train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
#     #     test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0, num_workers=4)
#     #     # test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=True)
#     else:
#         print("Preparing non-cached dataset...")
#         train_ds = Dataset(data=train_files, transform=train_transforms)
#         # train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
#         test_ds = Dataset(data=test_files, transform=test_transforms)
#         # test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=True)

#     print("Creating data loaders...")
#     train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
#     test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=True)

#     print(f"Number of training samples: {len(train_ds)}")
#     print(f"Number of test samples: {len(test_ds)}")

#     return train_loader, test_loader



# if __name__ == '__main__':
#     data_dir = '/home/juval.gutknecht/Projects/Data/USB'
#     result = prepare(data_dir, cache=True)
#     print("After calling prepare function:")
#     print(f"Result type: {type(result)}")
#     if isinstance(result, tuple) and len(result) == 2:
#         train_loader, test_loader = result
#         print(f"train_loader type: {type(train_loader)}")
#         print(f"test_loader type: {type(test_loader)}")
#     else:
#         print("prepare function did not return expected tuple")
#     print('Data prepared')