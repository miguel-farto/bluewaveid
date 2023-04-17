# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 09:41:30 2021

@author: user01
"""
import pandas as pd
import numpy as np
from skimage import io
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

def augmentation(image_size, train=True):
    max_crop = image_size // 5
    if train:
        data_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Compose(
                [
                    A.OneOf([
                        A.RandomRain(p=0.1),
                        A.GaussNoise(mean=15.),
                        A.GaussianBlur(blur_limit=9, p=0.4),
                        A.MotionBlur(p=0.2)
                    ]),

                    A.OneOf([
                        A.RandomBrightnessContrast(
                            brightness_limit=0.3, contrast_limit=0.1, p=1),
                        A.HueSaturationValue(hue_shift_limit=20, p=1),
                    ], p=0.6),

                    A.OneOf([
                        A.CLAHE(clip_limit=2.),
                        A.Sharpen(),
                        A.Emboss(),
                    ]),

                    A.OneOf([
                        A.Perspective(p=0.3),
                        A.ElasticTransform(p=0.1)
                    ]),

                    A.OneOf([
                        A.Rotate(limit=25, p=0.6),
                        A.Affine(
                            scale=0.9,
                            translate_px=15,
                            rotate=25,
                            shear=0.2,
                        )
                    ], p=1),

                    A.Cutout(num_holes=1, max_h_size=max_crop, max_w_size=max_crop, p=0.2)],
                p=1
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    return data_transform