import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from scipy.io import loadmat
from utils import get_density_map_gaussian

from transforms import *

class CrowdDataSet(Dataset):
    """Crowd-Couting dataset Loader"""

    def __init__(self, dirname, transform=None):
        """
        Parameters
        ----------
        dirname: Location of image folder (i.e /path/to/train, /path/to/val)
        transform: List of Transforms (default: None)
        debug: Prints out additional message in debug mode (default: False)
        sample_images: Use the List of images instead of reading from dirname
        """
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
        k = get_density_map_gaussian(k, gt, adaptive_mode=True)
        sample = {'image': image, 'den': k, 'gt': gt, 'fname': image_fname}

        if self.transform:
            sample = self.transform(sample)

        return sample



def default_train_transforms(output_size=448, factor=4):
    """Training set transforms

    Parameters
    ----------
    output_size: Resize the input image into square(width:output_size, height:output_size)
    factor: Scale down factor to apply on input image (default: 4)
    """
    return transforms.Compose([
        CenterCrop(output_size=output_size),
        RandomFlip(),
        ScaleDown(factor),
        LabelNormalize(),
        ToTensor(),
        Normalize()
    ])


def default_val_transforms(output_size=448, factor=4):
    """Validation set transforms

    Parameters
    ----------
    output_size: Resize the input image into square(width:output_size, height:output_size)
    factor: Scale down factor to apply on input image (default: 4)
    """
    return transforms.Compose([
        CenterCrop(output_size=output_size),
        ScaleDown(factor),
        LabelNormalize(),
        ToTensor(),
        Normalize()
    ])


def default_test_transforms():
    """Test set transforms"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.410824894905, 0.370634973049, 0.359682112932],
            [0.278580576181, 0.26925137639, 0.27156367898]
        )
    ])


def display_transforms():
    """Transforms to conver the tensor back to PIL Image"""
    return transforms.Compose([
        DeNormalize(),
        transforms.ToPILImage()
    ])