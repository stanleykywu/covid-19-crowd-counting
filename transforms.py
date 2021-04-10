import random
import numpy as np
import cv2
from PIL import ImageOps
from torchvision import transforms

class CenterCrop(object):
    def __init__(self, output_size=224, bins=False):
        self.output_size = output_size
        self.bins = bins

    def __call__(self, sample):
        image, den, gt, fname = sample['image'], sample['den'], sample['gt'], sample['fname']
        output_w, output_h = self.output_size, self.output_size
        # Original image size
        # Crop Image
        w, h = image.size
        xi1 = int((w - self.output_size) / 2)
        yi1 = int((h - self.output_size) / 2)
        cropped_img = image.crop((xi1, yi1, xi1 + self.output_size, yi1 + self.output_size))

        # Crop density (argh the code is so messy)
        output_w = min(w, output_w)
        output_h = min(h, output_h)
        x1 = int((w - output_w) / 2)
        y1 = int((h - output_h) / 2)
        new_gt = []

        # Crop ground truths
        for p in gt:
            x = p[0]
            y = p[1]
            if x > x1 and x < x1 + output_w and y > y1 and y < y1 + output_h:
                new_gt.append([x - x1, y - y1])

        if self.bins:
            return {"image": cropped_img, "den": None, "gt": new_gt, 'fname': fname}
            
        cropped_den = den[y1:y1 + output_h, x1:x1 + output_w]
        frame = np.full((self.output_size, self.output_size), 0.0)
        fx = int((self.output_size - cropped_den.shape[0]) / 2)
        fy = int((self.output_size - cropped_den.shape[1]) / 2)     
        frame[fx:fx+cropped_den.shape[0], fy:fy+cropped_den.shape[1]] = cropped_den[0:cropped_den.shape[0], 0:cropped_den.shape[1]]

        return {"image": cropped_img, "den": frame.astype('float32'), "gt": new_gt, 'fname': fname}



class RandomFlip(object):
    def __init__(self, bins=False):
        self.bins = bins

    def __call__(self, sample):
        image, den = sample['image'], sample['den']

        if self.bins:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            return {"image": image, "den": None, "gt": sample['gt'], 'fname': sample['fname']}
            
        # flip with 50% probability
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
            den = den[:, ::-1].copy()

        return {"image": image, "den": den, "gt": sample['gt'], 'fname': sample['fname']}


class ToTensor(object):
    """Convert Image and density map to tensor"""
    def __call__(self, sample):
        image = sample['image']

        image = transforms.Compose([
            transforms.ToTensor()
        ])(image)

        return {"image": image, "den": sample['den'], "gt": sample['gt'], 'fname': sample['fname']}


class Normalize(object):
    """Normalize Image"""
    def __init__(self, bins=False, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
        self.bins = bins

    def __call__(self, sample):
        image, den = sample['image'], sample['den']

        image = transforms.Compose([
            transforms.Normalize(self.mean, self.std)
        ])(image)

        if self.bins:
            return {"image": image, "den": den, "gt": sample['gt'], 'fname': sample['fname']}

        den = np.reshape(den, [1, *den.shape])
        return {"image": image, "den": den, "gt": sample['gt'], 'fname': sample['fname']}



class LabelNormalize(object):
    """Normalize the density map
    C3 Paper suggests that network converges faster when we use
    a large number in density map (they suggest 100)
    """
    def __init__(self):
        return

    def __call__(self, sample):
        return {"image": sample['image'], "den": sample['den'] * 100, "gt": sample['gt'], 'fname': sample['fname']}
