# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 09:06:11 2021

@author: user01
"""
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


class whalesDataset(Dataset):
    """Whales Kaggle dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file).Id_int
        #self.labels = np.array(pd.factorize(self.labels)[0])
        self.images = np.array(pd.read_csv(csv_file).Image)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.images[idx])
        #image = io.imread(img_name, as_gray=True)
        image = Image.open(img_name).convert('RGB') #.convert('L')
        image = np.array(image)
        #image = np.stack((image,)*3, axis=-1)
        image = image.astype('uint8')

        if type(self.transform) == A.core.composition.Compose:
            image = self.transform(image = image)["image"]
        else:
            image = self.transform(image)
        
        #image = np.expand_dims(image, 0)
        #image = image.astype('double')

        labels = self.labels[idx]
        labels = labels.astype('float')
        sample = {'image': image, 'label': labels}
        #sample = (image, labels)



        return sample