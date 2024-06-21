# Import
import os
import torch
import monai
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord, 
                              Activations, AsDiscrete, EnsureType)
from monai.data import decollate_batch
from monai.networks.nets import DynUNet
from monai.inferers import sliding_window_inference
from monai.config import print_config

# Checking MONAI configuration
print_config()

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


# Transformations
label_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
common_transforms = [
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityd(keys=["image"]),
    ToTensord(keys=["image"]),
]

inference_transforms = Compose(common_transforms)

# Load the trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DynUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=5,
    kernel_size=(3, 3, 3),
    strides=(1, 2, 2),
    upsample_kernel_size=(2, 2, 2)
).to(device)
model.load_state_dict(torch.load("best_metric_model.pth"))
model.eval()

# Define post-processing
post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

# Run inference function
def run_inference(image_path):
    data = {"image": image_path}
    input_image = inference_transforms(data)["image"].unsqueeze(0)  # Add batch dimension
    input_image = input_image.to(device)

    # Perform inference
    roi_size = (128, 128, 128)
    sw_batch_size = 4
    with torch.no_grad():
        pred = sliding_window_inference(input_image, roi_size, sw_batch_size, model)
        pred = post_trans(pred)

    return pred

# Example usage
image_path = "path_to_your_inference_image.nrrd"
predicted_output = run_inference(image_path)

# Converting the tensor to a numpy array for further processing or visualization
predicted_output_np = predicted_output[0].cpu().numpy()

# Visualize the result
plt.imshow(predicted_output_np[0, :, :, 64], cmap="gray")  # Adjust dimensions/slicing based on your data orientation
plt.title("Predicted Segmentation")
plt.show()

# Save the output as a NIfTI file
output_image = nib.Nifti1Image(predicted_output_np, np.eye(4))
nib.save(output_image, "output_segmentation.nii.gz")