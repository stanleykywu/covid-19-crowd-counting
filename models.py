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
    def __init__(self, channels=[512, 128, 1], scale_factor=4):
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

class ResNet18Classification(nn.Module):
    def __init__(self, bins=5):
        """
        Parameters
        ----------
        channels: Input channel size for all three layers
        scale_factor: Factor to upsample the feature map
        """
        super(ResNet18Classification, self).__init__()
        self.bins = bins
        
        
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        
        model_ft.fc = nn.Sequential(
            nn.Linear(in_features=num_ftrs, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=self.bins, bias=True),
            nn.LogSoftmax()
        )

        self.model = model_ft
    
    def forward(self, inputs):
        return self.model(inputs)