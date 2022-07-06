import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import neptune.new as neptune
from tqdm import tqdm

from transformer import VisionTransformer


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

    best_accuracy = 0
    for epoch in range(1, n_epochs + 1):
        print('Epoch:', epoch)
        for i, (data, target) in tqdm(enumerate(train_data), total=len(train_data), desc="Training"):
            total_samples = len(train_data.dataset)
            best_accuracy = 0
            model.train()
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
                total_loss += val_loss.item()
                correct_samples += pred.eq(y).sum()
                val_acc = torch.sum(pred == y).item()/len(val_data)
                avg_loss = total_loss / total_samples

                if monitoring:
                    run['Val_loss'].log(avg_loss)
                    run['Val_accuracy'].log(val_acc)
    

if __name__ == '__main__':
    from preprocessing import Dataset 
    from config import Config
    config = Config()

    model = VisionTransformer(img_size=config.IMG_SIZE,
                 num_classes=config.NUM_CLASSES,
                 hidden_size=config.HIDDEN_SIZE,
                 in_channels=config.IN_CHANNELS,
                 num_layers=config.NUM_LAYERS,
                 num_attention_heads=config.NUM_ATTENTION_HEADS,
                 linear_dim=config.LINEAR_DIM,
                 dropout_rate=config.DROPOUT_RATE,
                 attention_dropout_rate=config.ATTENTION_DROPOUT_RATE,
                 eps=config.EPS,
                 std_norm=config.STD_NORM)

    train_data, val_data, test_data = Dataset(config.BATCH_SIZE,
                                              config.IMG_SIZE, 
                                              config.DATASET_SAMPLE)# neptune.save_checkpoint(
                                                
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    train_Engine(n_epochs=config.N_EPOCHS,
                train_data=train_data,
                val_data=val_data,
                model=model,
                optimizer=optimizer,
                loss_fn='nll_loss',
                device=config.DEVICE[1], monitoring=False)                              
    print("done")