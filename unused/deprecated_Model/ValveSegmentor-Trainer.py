# Import
import os
import glob
import torch
import monai
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord, 
                              AsDiscrete, AsDiscreted, Activations, Activationsd, EnsureType, EnsureTyped, 
                              RandAffined, RandRotate90d, RandFlipd, RandZoomd, SpatialPadd, CenterSpatialCropd, MapTransform)
from monai.data import CacheDataset, SmartCacheDataset, DataLoader
from monai.networks.nets import DynUNet
from monai.losses import DiceLoss as MonaiDiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast


class RemapLabels(monai.transforms.MapTransform):
    def __init__(self, keys, mapping):
        super().__init__(keys)
        self.mapping = mapping

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            for old_val, new_val in self.mapping.items():
                d[key][d[key] == old_val] = new_val
        return d


# Set CUDA_LAUNCH_BLOCKING for debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Data Preparation
data_dir = "/home/juval.gutknecht/Projects/Heart-Valve-Segmentation/Data"
data_folders = [os.path.join(data_dir, folder) for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

image_files = []
label_files = []

for folder in data_folders:
    image_files.extend(glob.glob(os.path.join(folder, "*.nrrd")))
    label_files.extend(glob.glob(os.path.join(folder, "*.seg.nrrd")))

# Create data dictionaries
data_files = [{"image": img, "label": lbl} for img, lbl in zip(image_files, label_files)]
train_files, val_files = train_test_split(data_files, test_size=0.2, random_state=42) # Change test_size to 0.2 for validation split



# Define transformations
label_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

common_transforms = [
    LoadImaged(keys=["image", "label"]), 
    EnsureChannelFirstd(keys=["image", "label"]), 
    ScaleIntensityd(keys=["image"]), 
    RemapLabels(keys=["label"], mapping=label_mapping)
                    ]

train_transforms = Compose(
    common_transforms + [
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]), 
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5), 
        RandZoomd(keys=["image", "label"], min_zoom=0.9, max_zoom=1.1, prob=0.5), 
        RandAffined(keys=["image", "label"], prob=0.5, rotate_range=(0, 0.3), scale_range=(0.8, 1.2), mode='bilinear'), 
        SpatialPadd(keys=["image", "label"], spatial_size=[160, 160, 160], mode='constant'), 
        CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 128]), 
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["label"]), 
        # AsDiscrete(keys=["label"], argmax=True) 
        ])

val_transforms = Compose(
    common_transforms + [
        SpatialPadd(keys=["image", "label"], spatial_size=[160, 160, 160], mode='constant'), 
        CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 128]), 
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["label"]), 
        # AsDiscrete(keys=["label"], argmax=True) 
        ])


# Post processing
post_trans = Compose([
    EnsureType(),
    Activations(sigmoid=True), 
    AsDiscrete(keys="pred", argmax=True, threshold=0.5)
    ])


# Data Loaders
train_dataset = SmartCacheDataset(data=train_files, transform=train_transforms, cache_num=64, cache_rate=0.1, replace_rate=0.2)
val_dataset = SmartCacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)


# Model, Loss, Optimizer, Metrics
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = DynUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=5,
    kernel_size=(3, 3, 3),
    strides=(1, 2, 2),
    upsample_kernel_size=(2, 2, 2),
).to(device)

loss_function = MonaiDiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")
scaler = GradScaler()
writer = SummaryWriter()

# Training and Validation Loop
max_epochs = 100
val_interval = 2
best_metric = -1
best_metric_epoch = -1

for epoch in range(max_epochs):
    print("-" * 10) 
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        
        # # Debugging prints
        # print(f"Input shape: {inputs.shape}, dtype: {inputs.dtype}")
        # print(f"Label shape: {labels.shape}, dtype: {labels.dtype}")
        # print(f"Unique label values: {torch.unique(labels)}")

        assert inputs.shape == labels.shape, "Shape mismatch between inputs and labels"
        assert inputs.dtype == torch.float32, "Input tensor dtype is not float32"
        assert labels.dtype == torch.float32, "Label tensor dtype is not float32"

        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        epoch_len = len(train_loader.dataset) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
    epoch_loss /= step
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    writer.add_scalar("train_loss", epoch_loss, epoch + 1)

    if (epoch + 1) % val_interval == 0:
        model.eval()
        
        with torch.no_grad():
            val_images = None
            val_labels = None
            val_outputs = None
            for val_data in val_loader:
                val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                roi_size = (128, 128, 128)
                sw_batch_size = 1
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                val_outputs = post_trans(val_outputs)
                val_labels = post_trans(val_labels)

                print(f"val_outputs shape: {val_outputs.shape}, dtype: {val_outputs.dtype}")
                print(f"val_labels shape: {val_labels.shape}, dtype: {val_labels.dtype}")
                print(f"Unique label values in predictions: {torch.unique(val_outputs)}")
                print(f"Unique label values in labels: {torch.unique(val_labels)}")
                
                dice_metric(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            print(f"Validation - Mean Dice: {metric:.4f}")

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model.pth")
                print("saved new best metric model")
            print(f"current epoch: {epoch + 1} current mean dice: {metric:.4f} best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")
            writer.add_scalar("val_mean_dice", metric, epoch + 1)
            dice_metric.reset()

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()


