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
                 monitoring=True):

    logits_m = []
    yo = []
    attention = []
    confusion_matrix = torch.zeros(400, 400)
    best_accuracy = 0
    for epoch in range(1, n_epochs + 1):
        print('Epoch:', epoch)
        for i, (data, target) in tqdm(enumerate(train_data), total=len(train_data), desc="Training"):
            total_samples = len(train_data.dataset)
            best_accuracy = 0
            #device
            model = model.to(device)
            x = data.to(device)
            y = target.to(device)
            optimizer.zero_grad()
            
            logits, attn_weights = model(x)
            proba = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(proba, y, reduction='sum')
            loss.backward()
            
            yhat = torch.argmax(logits, dim=1)
            accuracy = torch.sum(yhat == y).item()/len(train_data)            
            optimizer.step()
            
            if monitoring:
                run['Training_loss'].log(loss.item())
                run['Training_acc'].log(accuracy)

            if accuracy > best_accuracy:
                best_model = model
        torch.save(best_model.state_dict(), 'model.pt')
        model.eval()
        total_samples = len(val_data.dataset)
        correct_samples = 0
        total_loss = 0
        
        with torch.no_grad():
            for i, (data, target) in tqdm(enumerate(val_data), total=len(val_data), desc="Evaluation"):
                
                model = model.to(device)
                x = data.to(device)
                y = target.to(device)
            
                logits, attn_weights = model(x)
                proba = F.log_softmax(logits, dim=1)
                val_loss = F.nll_loss(proba, y, reduction='sum')
                _, pred = torch.max(logits, dim=1)
                
                logits_m.append(logits)
                yo.append(y)
                attention.append(attn_weights)
                for t, p in zip(y.view(-1), pred.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

                
                total_loss += val_loss.item()
                correct_samples += pred.eq(y).sum()
                val_acc = torch.sum(pred == y).item()/len(val_data)
                avg_loss = total_loss / total_samples

                if monitoring:
                    run['Val_loss'].log(avg_loss)
                    run['Val_accuracy'].log(val_acc)
    
    return confusion_matrix, logits_m, yo, attention

    # neptune.save_checkpoint(model, optimizer, epoch=N_EPOCHS)