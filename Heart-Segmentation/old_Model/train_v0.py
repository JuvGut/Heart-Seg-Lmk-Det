from monai.networks.nets import DynUNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss
import os
import torch
from preporcess_v0 import prepare
from utilities_v0 import train, show_patient


data_dir = '/home/juval.gutknecht/Projects/Data/USB'
model_dir = '/home/juval.gutknecht/Projects/RESULTS' 

os.makedirs(model_dir, exist_ok=True)
data_in = prepare(data_dir, cache=True)

device = torch.device("cuda:2")
model = DynUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=5,
    kernel_size=(3, 3, 3),
    strides=(1, 2, 2),
    upsample_kernel_size=(2, 2),
    filters=(16, 32, 64),
    norm_name=Norm.BATCH,
).to(device)


#loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=calculate_weights(1792651250,2510860).to(device))
loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)

if __name__ == '__main__':
    #show_patient(data_in)
    train(model, data_in, loss_function, optimizer, 600, model_dir)