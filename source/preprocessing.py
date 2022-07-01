import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import cv2
import numpy as np
import matplotlib.pyplot as plt

import os

from config import Config


train_dir = '/Users/nielspace/Documents/data/birds/train/'
val_dir = '/Users/nielspace/Documents/data/birds/valid/'
test_dir = '/Users/nielspace/Documents/data/birds/test/'

def dataset(bs, crop_size):
    transformations = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(Config.CROP_SIZE), 
        torchvision.transforms.CenterCrop(Config.CROP_SIZE),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = torchvision.datasets.ImageFolder(root=train_dir, transform=transformations)
    test_data = torchvision.datasets.ImageFolder(root=test_dir, transform=transformations)
    valid_data = torchvision.datasets.ImageFolder(root=val_dir, transform=transformations)
    
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=Config.BATCH_SIZE)
    val_loader = torch.utils.data.DataLoader(valid_data, shuffle=True, batch_size=Config.BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=Config.BATCH_SIZE)

    
    return train_loader, val_loader, test_loader


