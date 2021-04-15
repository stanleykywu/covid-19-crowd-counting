import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from scipy.io import loadmat
from utils import get_density_map_gaussian

from transforms import *

class CrowdDataSet(Dataset):
    def __init__(self, dirname, transform=None):
        self.dirname = dirname
        self.images = os.listdir(dirname + '/images')
        self.length = len(self.images)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_fname = self.images[idx]
        image = Image.open(self.dirname + '/images/' + image_fname)
        image = image.convert('RGB')
        pts = loadmat((self.dirname + '/ground-truth/' + image_fname).replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG_', 'GT_IMG_'))
        gt = pts["image_info"][0, 0][0, 0][0]

        k = np.zeros((image.width, image.height))
        k = get_density_map_gaussian(k, gt, adaptive_mode=False)
        sample = {'image': image, 'den': k, 'gt': gt, 'fname': image_fname}

        if self.transform:
            sample = self.transform(sample)

        return sample

class CrowdClassificationDataSet(Dataset):
    def __init__(self, dirname, transform=None):
        self.dirname = dirname
        self.images = os.listdir(dirname + '/images')
        self.length = len(self.images)
        self.transform = transform
        self.bins = [  0.        ,  20.        ,  42.60000153,  71.79999847,
        129.2       , 804.99993896]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_fname = self.images[idx]
        image = Image.open(self.dirname + '/images/' + image_fname)
        image = image.convert('RGB')
        pts = loadmat((self.dirname + '/ground-truth/' + image_fname).replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG_', 'GT_IMG_'))
        gt = pts["image_info"][0, 0][0, 0][0]

        sample = {'image': image, 'den': None, 'gt': gt, 'fname': image_fname}

        if self.transform:
            sample = self.transform(sample)

        k = np.zeros((224, 224))
        k = get_density_map_gaussian(k, sample['gt'], adaptive_mode=False)
        num = np.sum(k)
        category = 4
        for i, bin in enumerate(self.bins):
            if num == 0:
                category = 0
                break
            if num <= bin:
                category = i - 1
                break
        sample = {'image': sample['image'], 'bin': category, 'num': num, 'fname': image_fname}

        return sample


def default_train_transforms(output_size=224):
    return transforms.Compose([
        CenterCrop(output_size=output_size),
        RandomFlip(),
        LabelNormalize(),
        ToTensor(),
        Normalize()
    ])

def default_val_transforms(output_size=224):
    return transforms.Compose([
        CenterCrop(output_size=output_size),
        LabelNormalize(),
        ToTensor(),
        Normalize()
    ])

def default_train_transform_classification(output_size=224):
    return transforms.Compose([
        CenterCrop(output_size=output_size, bins=True),
        RandomFlip(bins=True),
        ToTensor(),
        Normalize(bins=True)
    ])

def default_val_transform_classification(output_size=224):
    return transforms.Compose([
        CenterCrop(output_size=output_size, bins=True),
        ToTensor(),
        Normalize(bins=True)
    ])

