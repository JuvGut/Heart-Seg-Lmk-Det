from monai.utils import first
import matplotlib.pyplot as plt
import torch
import os
from datetime import datetime
import numpy as np
from monai.losses import DiceLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def dice_metric(predicted, target):
    '''
    In this function we take `predicted` and `target` (label) to calculate the dice coeficient then we use it 
    to calculate a metric value for the training and the validation.
    '''
    dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    value = 1 - dice_value(predicted, target).item()
    return value

def calculate_weights(val1, val2):
    '''
    In this function we take the number of the background and the forgroud pixels to return the `weights` 
    for the cross entropy loss values.
    '''
    count = np.array([val1, val2])
    summ = count.sum()
    weights = count/summ
    weights = 1/weights
    summ = weights.sum()
    weights = weights/summ
    return torch.tensor(weights, dtype=torch.float32)

def train(model, data_in, loss, optim, max_epochs, model_dir, test_interval=1 , device=torch.device("cuda:2")):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(model_dir, 'logs', current_time)
    writer = SummaryWriter(log_dir=log_dir)
    
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_test = []
    save_metric_train = []
    save_metric_test = []
    train_loader, test_loader = data_in


    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train = 0
        for batch_data in train_loader:
            
            train_step += 1

            volume = batch_data["vol"]
            label = batch_data["seg"]
            label = label != 0
            volume, label = (volume.to(device), label.to(device))

            optim.zero_grad()
            outputs = model(volume)
            
            train_loss = loss(outputs, label)
            
            train_loss.backward()
            optim.step()

            train_epoch_loss += train_loss.item()
            train_metric = dice_metric(outputs, label)
            epoch_metric_train += train_metric
            
        train_epoch_loss /= train_step
        epoch_metric_train /= train_step

        print(f'Epoch_loss: {train_epoch_loss:.4f}')
        print(f'Epoch_metric: {epoch_metric_train:.4f}')
        
        save_loss_train.append(train_epoch_loss)
        save_metric_train.append(epoch_metric_train)
        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)
        np.save(os.path.join(model_dir, 'metric_train.npy'), save_metric_train)
    
        writer.add_scalar('Train_Loss', train_epoch_loss, epoch)
        writer.add_scalar('Train_Dice', epoch_metric_train, epoch)
            
        print('-'*20)
        
        if (epoch + 1) % test_interval == 0:
            model.eval()
            with torch.no_grad():
                test_epoch_loss = 0
                test_metric = 0
                epoch_metric_test = 0
                test_step = 0

                for test_data in test_loader:
                    test_step += 1

                    test_volume = test_data["vol"]
                    test_label = test_data["seg"]
                    test_label = test_label != 0
                    test_volume, test_label = (test_volume.to(device), test_label.to(device),)
                    
                    test_outputs = model(test_volume)
                    
                    test_loss = loss(test_outputs, test_label)
                    test_epoch_loss += test_loss.item()
                    test_metric = dice_metric(test_outputs, test_label)
                    epoch_metric_test += test_metric
                    
                test_epoch_loss /= test_step
                epoch_metric_test /= test_step
                
                print(f'test_loss_epoch: {test_epoch_loss:.4f}')
                print(f'test_dice_epoch: {epoch_metric_test:.4f}')

                save_loss_test.append(test_epoch_loss)
                save_metric_test.append(epoch_metric_test)
                np.save(os.path.join(model_dir, 'loss_test.npy'), save_loss_test)
                np.save(os.path.join(model_dir, 'metric_test.npy'), save_metric_test)

                writer.add_scalar('Loss/test', test_epoch_loss, epoch)
                writer.add_scalar('Dice/test', epoch_metric_test, epoch)

                if epoch_metric_test > best_metric:
                    best_metric = epoch_metric_test
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        model_dir, "best_metric_model.pth"))
                
                print(
                    f"Current epoch: {epoch + 1}, Current mean dice: {test_metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                )
    
    writer.close()
    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    
    return save_loss_train, save_loss_test, save_metric_train, save_metric_test

def show_patient(data, SLICE_NUMBER=1, train=True, test=False):
    """
    This function is to show one patient from your datasets, so that you can see if the it is okay or you need 
    to change/delete something.

    `data`: this parameter should take the patients from the data loader, which means you need to can the function
    prepare first and apply the transforms that you want after that pass it to this function so that you visualize 
    the patient with the transforms that you want.
    `SLICE_NUMBER`: this parameter will take the slice number that you want to display/show
    `train`: this parameter is to say that you want to display a patient from the training data (by default it is true)
    `test`: this parameter is to say that you want to display a patient from the testing patients.
    """

    check_patient_train, check_patient_test = data

    view_train_patient = first(check_patient_train)
    view_test_patient = first(check_patient_test)

    
    if train:
        plt.figure("Visualization Train", (18, 6))
        plt.subplot(1, 6, 1)
        plt.title(f"vol {SLICE_NUMBER}")
        plt.imshow(view_train_patient["vol"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        for i in range(5):
            plt.subplot(1, 6, i+2)
            plt.title(f"seg {i} {SLICE_NUMBER}")
            plt.imshow(view_train_patient["seg"][0, i, :, :, SLICE_NUMBER])
        plt.show()
    
    if test:
        plt.figure("Visualization Test", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"vol {SLICE_NUMBER}")
        plt.imshow(view_test_patient["vol"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        for i in range(5):
            plt.subplot(1, 6, i+2)
            plt.title(f"seg {i} {SLICE_NUMBER}")
            plt.imshow(view_test_patient["seg"][0, i, :, :, SLICE_NUMBER])
        plt.show()
    

def calculate_pixels(data):
    val = np.zeros((1, 2))

    for batch in tqdm(data):
        batch_label = batch["seg"] != 0
        _, count = np.unique(batch_label, return_counts=True)

        if len(count) == 1:
            count = np.append(count, 0)
        val += count

    print('The last values:', val)
    return val
