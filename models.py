import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGG16Transfer(nn.Module):
    def __init__(self, channels=[512, 128, 1], scale_factor=4):
        """
        Parameters
        ----------
        channels: Input channel size for all three layers
        scale_factor: Factor to upsample the feature map
        """
        super(VGG16Transfer, self).__init__()
        self.scale_factor = scale_factor
        
        
        conv_layers = list(models.vgg16(pretrained=True).features.children())
        # Mark the backbone as not trainable
        for layer in conv_layers:
            layer.requires_grad = False

        self.model = nn.Sequential(
            *conv_layers,
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    
    def forward(self, inputs):
        """ Forward pass

        Parameters
        ----------
        inputs: Batch of input images
        """
        output = self.model(inputs)
        output = F.upsample(output, scale_factor=self.scale_factor)
        return output


class ResNetTransfer(nn.Module):
    def __init__(self, channels=[512, 384, 256, 192, 128, 64, 1], scale_factor=4):
        """
        Parameters
        ----------
        channels: Input channel size for all three layers
        scale_factor: Factor to upsample the feature map
        """
        super(ResNetTransfer, self).__init__()
        self.scale_factor = scale_factor
        
        
        conv_layers = list(models.resnet18(pretrained=True).children())[0:8]
        # Mark the backbone as not trainable
        for layer in conv_layers:
            layer.requires_grad = False

        self.model = nn.Sequential(
            *conv_layers,
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[3], channels[4], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[4], channels[5], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[5], channels[6], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    
    def forward(self, inputs):
        """ Forward pass

        Parameters
        ----------
        inputs: Batch of input images
        """
        output = self.model(inputs)
        output = F.upsample(output, scale_factor=self.scale_factor)
        return output

class ResNetClassification(nn.Module):
    def __init__(self, bins=5):
        super(ResNetClassification, self).__init__()
        self.bins = bins
        
        model_ft = models.resnext50_32x4d(pretrained=True)

        for param in model_ft.parameters():
            param.requires_grad = False

        num_ftrs = model_ft.fc.in_features
        
        model_ft.fc = nn.Sequential(
            nn.Linear(in_features=num_ftrs, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=self.bins, bias=True),
        )

        self.model = model_ft
    
    def forward(self, inputs):
        return self.model(inputs)

class VGG16Classification(nn.Module):
    def __init__(self, bins=5):
        super(VGG16Classification, self).__init__()
        self.bins = bins
        
        model_ft = models.vgg16(pretrained=True)
        num_ftrs = 4096

        for param in model_ft.parameters():
            param.requires_grad = False
        
        model_ft.classifier[6] = nn.Sequential(
            nn.Linear(in_features=num_ftrs, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=self.bins, bias=True),
        )

        for param in model_ft.classifier[6].parameters():
            param.requires_grad = True

        self.model = model_ft
    
    def forward(self, inputs):
        return self.model(inputs)