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


train_dir = "/Users/nielspace/Documents/data/birds/train/"
val_dir = "/Users/nielspace/Documents/data/birds/valid/"
test_dir = "/Users/nielspace/Documents/data/birds/test/"


def Dataset(bs, crop_size, sample_size="full"):
    transformations = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(crop_size),
            torchvision.transforms.CenterCrop(crop_size),
            torchvision.transforms.horizontal_flip(),
            torchvision.transforms.RandomRotation(90),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    if sample_size == "full":
        train_data = torchvision.datasets.ImageFolder(
            root=train_dir, transform=transformations
        )
        valid_data = torchvision.datasets.ImageFolder(
            root=val_dir, transform=transformations
        )
        test_data = torchvision.datasets.ImageFolder(
            root=test_dir, transform=transformations
        )

        train_data = torch.utils.data.DataLoader(
            train_data, shuffle=True, batch_size=bs
        )
        valid_data = torch.utils.data.DataLoader(
            valid_data, shuffle=True, batch_size=bs
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, shuffle=True, batch_size=bs
        )

    else:
        train_data = torchvision.datasets.ImageFolder(
            root=train_dir, transform=transformations
        )
        indices = torch.arange(sample_size)
        train_data = torch.utils.data.Subset(train_data, indices)
        train_data = torch.utils.data.DataLoader(
            train_data, shuffle=True, batch_size=bs
        )

    valid_data = torchvision.datasets.ImageFolder(
        root=val_dir, transform=transformations
    )
    valid_data = torch.utils.data.DataLoader(valid_data, shuffle=True, batch_size=bs)

    test_data = torchvision.datasets.ImageFolder(
        root=test_dir, transform=transformations
    )
    test_data = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=bs)

    return train_data, valid_data, test_data
