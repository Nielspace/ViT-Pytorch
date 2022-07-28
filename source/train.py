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


def neptune_monitoring(config):
    PARAMS = {}
    for key, val in config.__dict__.items():
        if key not in ["__module__", "__dict__", "__weakref__", "__doc__"]:
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
    
    train_accuracy = 0
    val_accuracy = 0
    best_accuracy = 0
    for epoch in range(1, n_epochs + 1):
        total = 0
        with tqdm(train_data, unit="iteration") as train_epoch:
            train_epoch.set_description(f"Epoch {epoch}")
            for i, (data, target) in enumerate(train_epoch):
                total_samples = len(train_data.dataset)
                #device
                model = model.to(device)
                x = data.to(device)
                y = target.to(device)
                optimizer.zero_grad()

                logits, attn_weights = model(x)
                proba = F.log_softmax(logits, dim=1)
                loss = F.nll_loss(proba, y, reduction='sum')
                loss.backward()           
                optimizer.step()
        
                _, pred = torch.max(logits, dim=1) #
                train_accuracy += torch.sum(pred==y).item()
                total += target.size(0)
                accuracy_=(100 *  train_accuracy/ total)
                train_epoch.set_postfix(loss=loss.item(), accuracy=accuracy_)
                
                if monitoring:
                    run['Training_loss'].log(loss.item())
                    run['Training_acc'].log(accuracy_)

                if accuracy_ > best_accuracy:
                    best_accuracy = accuracy_
                    best_model = model
                    torch.save(best_model, f'/metadata/model.pth')
                
        
        total_samples = len(val_data.dataset)
        correct_samples = 0
        total_ = 0
        model.eval()
        with torch.no_grad():
            with tqdm(val_data, unit="iteration") as val_epoch:
                val_epoch.set_description(f"Epoch {epoch}")
                for i, (data, target) in enumerate(val_epoch):
                    
                    model = model.to(device)
                    x = data.to(device)
                    y = target.to(device)
                    
                    logits,attn_weights = model(x)
                    proba = F.log_softmax(logits, dim=1)
                    val_loss = F.nll_loss(proba, y, reduction='sum')
                    
                    _, pred = torch.max(logits, dim=1)#
                    val_accuracy += torch.sum(pred==y).item()
                    total_ += target.size(0)
                    val_accuracy_ = (100 *  val_accuracy/ total_)
                    val_epoch.set_postfix(loss=val_loss.item(), accuracy=val_accuracy_)

                    if monitoring:
                        run['Val_accuracy '].log(val_accuracy_)
                        run['Val_loss'].log(loss.item())
    


if __name__ == "__main__":
    from preprocessing import Dataset
    from config import Config

    config = Config()
    params = neptune_monitoring(Config)
    run = neptune.init(
        project="nielspace/ViT-bird-classification",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkYjRhYzI0Ny0zZjBmLTQ3YjYtOTY0Yi05ZTQ4ODM3YzE0YWEifQ==",
    )
    run["parameters"] = params

    model = VisionTransformer(
        img_size=config.IMG_SIZE,
        num_classes=config.NUM_CLASSES,
        hidden_size=config.HIDDEN_SIZE,
        in_channels=config.IN_CHANNELS,
        num_layers=config.NUM_LAYERS,
        num_attention_heads=config.NUM_ATTENTION_HEADS,
        linear_dim=config.LINEAR_DIM,
        dropout_rate=config.DROPOUT_RATE,
        attention_dropout_rate=config.ATTENTION_DROPOUT_RATE,
        eps=config.EPS,
        std_norm=config.STD_NORM,
    )

    train_data, val_data, test_data = Dataset(
        config.BATCH_SIZE, config.IMG_SIZE, config.DATASET_SAMPLE
    )  # neptune.save_checkpoint(

    optimizer = optim.Adam(model.parameters(), lr=0.003)
    train_Engine(
        n_epochs=config.N_EPOCHS,
        train_data=train_data,
        val_data=val_data,
        model=model,
        optimizer=optimizer,
        loss_fn="nll_loss",
        device=config.DEVICE[1],
        monitoring=True,
    )

