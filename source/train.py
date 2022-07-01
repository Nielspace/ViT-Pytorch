import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
from tqdm import tqdm

import neptune.new as neptune

from config import Config
from preprocessing import dataset
from tqdm import tqdm
from transformer import ViT



IMG_SIZE = Config.IMG_SIZE
PATCH_SIZE = Config.PATCH_SIZE
CROP_SIZE = Config.CROP_SIZE
N_SAMPLES = Config.N_SAMPLES
BATCH_SIZE = Config.BATCH_SIZE
N_TRAIN = Config.N_TRAIN
N_VAL = Config.N_VAL
N_TEST = Config.N_TEST

LR = Config.LR
OPIMIZER = Config.OPIMIZER


N_CLASSES = Config.N_CLASSES
N_CHANNELS = Config.N_CHANNELS
N_DIM = Config.N_DIM
DEPTH = Config.DEPTH
HEADS = Config.HEADS
MPL_DIM = Config.MPL_DIM
OUTPUT = Config.OUTPUT
LOSS_FN = Config.LOSS_FN


DEVICE = Config.DEVICE[1]

N_EPOCHS = Config.N_EPOCHS
TRAIN_LOSS_HISTORY =  Config.TRAIN_LOSS_HISTORY
VAL_LOSS_HISTORY = Config.VAL_LOSS_HISTORY

model =  ViT(
    image_size=IMG_SIZE, 
    patch_size=PATCH_SIZE, 
    num_classes=N_CLASSES, 
    channels=N_CHANNELS, 
    dim=N_DIM, 
    depth=DEPTH, 
    heads=HEADS, 
    mlp_dim=MPL_DIM,
)


def neptune_monitoring():
    PARAMS = {}
    for key, val in Config.__dict__.items():
        if key not in ['__module__', '__dict__', '__weakref__', '__doc__']:
            PARAMS[key] = val 
    return PARAMS



def train_Engine(n_epochs,
                 train_data,
                 val_data,
                 model,
                 optimizer,
                 loss_fn,
                 device,
                 train_loss_history,
                 val_loss_history,
                 monitoring=True):

    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        for i, (data, target) in tqdm(enumerate(train_data), total=len(train_data), desc="Training"):
            total_samples = len(train_data.dataset)
            
            #device
            model = model.to(device)
            x = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = F.log_softmax(model.forward(x), dim=1)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            if monitoring:
                run['training_loss'].log(loss.item())


            if i % 100 == 0:
                print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                    ' (' + '{:3.0f}'.format(100 * i / len(train_data)) + '%)]  Loss: ' +
                    '{:6.4f}'.format(loss.item()))
                train_loss_history.append(loss.item())

            
        model.eval()
        total_samples = len(valid_data.dataset)
        correct_samples = 0
        total_loss = 0

        with torch.no_grad():
            for i, (data, target) in tqdm(enumerate(valid_data), total=len(train_data), desc="Valuation"):
                
                model = model.to(device)
                x = data.to(device)
                target = target.to(device)

                output = F.log_softmax(model(data), dim=1)
                val_loss = F.nll_loss(output, target, reduction='sum')
                _, pred = torch.max(output, dim=1)
                
                total_loss += val_loss.item()
                correct_samples += pred.eq(target).sum()
                
                avg_loss = total_loss / total_samples
                val_loss_history.append(avg_loss)

                if monitoring:
                    run['training_loss'].log(avg_loss)
                print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
                    '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
                    '{:5}'.format(total_samples) + ' (' +
                    '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')


if __name__ == '__main__':

    train_data, val_data, test_data = dataset(BATCH_SIZE, CROP_SIZE)
    params = neptune_monitoring()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    run = neptune.init(
        project="nielspace/ViT-bird-classification",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkYjRhYzI0Ny0zZjBmLTQ3YjYtOTY0Yi05ZTQ4ODM3YzE0YWEifQ==",
    )
    run["parameters"] = params
    
    train_Engine(n_epochs=N_EPOCHS,
                 train_data=train_data,
                 val_data=val_data,
                 model=model,
                 optimizer=optimizer,
                 loss_fn=LOSS_FN,
                 device=DEVICE,
                 train_loss_history=TRAIN_LOSS_HISTORY,
                 val_loss_history=VAL_LOSS_HISTORY,)

    neptune.save_checkpoint(model, optimizer, epoch=N_EPOCHS)