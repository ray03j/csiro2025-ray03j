from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
import os
import warnings
import cv2

class MyDataset(Dataset):
    def __init__(self, data, cfg, mode):
        assert mode in ["train", "val", "test"]
        self.cfg = cfg
        self.mode = mode
        self.transforms = get_train_transforms(cfg) if mode == "train" else get_val_transforms(cfg)
        self.data = data.reset_index(drop=True)
        self.vertical_split = True

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        image = item["image"]

        # 画像を左右に分割
        if self.vertical_split:
            width, height = image.size
            mid_point = width // 2
            left_image = image.crop((0, 0, mid_point, height))
            right_image = image.crop((mid_point, 0, width, height))

            if self.transforms:
                left_image = self.transforms(image=np.array(left_image))['image']
                right_image = self.transforms(image=np.array(right_image))['image']

            targets = torch.tensor(
                [
                    item["Dry_Green_g"], 
                    item["Dry_Dead_g"], 
                    item["Dry_Clover_g"], 
                    item["GDM_g"], 
                    item["Dry_Total_g"]
                ],
                dtype=torch.float32,
            )
            return left_image, right_image, targets

        else:
            if self.transforms:
                image = self.transforms(image=np.array(image))['image']
            targets = torch.tensor(
                [
                    item["Dry_Green_g"], 
                    item["Dry_Dead_g"], 
                    item["Dry_Clover_g"], 
                    item["GDM_g"], 
                    item["Dry_Total_g"]
                ],
                dtype=torch.float32,
            )
            return image, targets

def get_train_transforms(cfg):
    return A.Compose(
        [
            A.Resize(height=cfg.task.img_size, width=cfg.task.img_size, p=1),
            # A.XYMasking(num_masks_x=(1, 4), num_masks_y=(1, 4), mask_y_length=(0, 32), mask_x_length=(0, 32),
            #             fill_value=-1.0, p=0.5),
            # torchvision: ColorJitter
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
            # A.CenterCrop(height=cfg.task.img_size, width=cfg.task.img_size, p=1),
            # A.ShiftScaleRotate(p=0.5, border_mode=cv2.BORDER_CONSTANT, shift_limit=0.05, scale_limit=0.1, value=0,
            #                    rotate_limit=180, mask_value=0),
            # A.RandomScale(scale_limit=(0.8, 1.2), p=1),
            # A.PadIfNeeded(min_height=cfg.model.img_size, min_width=cfg.model.img_size, p=1.0,
            #              border_mode=cv2.BORDER_CONSTANT, value=0),
            # A.RandomCrop(height=self.cfg.data.train_img_h, width=self.cfg.data.train_img_w, p=1.0),
            # A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=0.5),
            A.HorizontalFlip(p=0.5), # torchvision: RandomHorizontalFlip
            A.VerticalFlip(p=0.5), # torchvision: RandomVerticalFlip
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=(-10, 10), p=0.3, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
            # A.RandomBrightnessContrast(p=0.5, brightness_limit=0.1, contrast_limit=0.1),
            # A.HueSaturationValue(p=0.5),
            # A.ToGray(p=0.3),
            # A.GaussNoise(var_limit=(0.0, 0.05), p=0.5),
            # A.GaussianBlur(p=0.5),
            # torchvision: Normalize(ImageNet)
            A.Normalize(p=1.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
            # A.RandomBrightnessContrast(
            #     brightness_limit=0.3, contrast_limit=0.3, p=0.3
            # ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )


def get_val_transforms(cfg):
    return A.Compose(
        [
            A.Resize(height=cfg.task.img_size, width=cfg.task.img_size, p=1),
            # A.RandomScale(scale_limit=(1.0, 1.0), p=1),
            # A.PadIfNeeded(min_height=cfg.task.img_size, min_width=cfg.task.img_size, p=1.0,
            #              border_mode=cv2.BORDER_CONSTANT, value=0),
            # A.Crop(y_max=self.cfg.data.val_img_h, x_max=self.cfg.data.val_img_w, p=1.0),
            A.Normalize(p=1.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )
