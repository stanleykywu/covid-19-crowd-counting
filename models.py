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
        
        
        conv_layers = list(models.resnet18(pretrained=True).features.children())
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


class InceptionV3Transfer(nn.Module):
    def __init__(self, channels=[512, 128, 1], scale_factor=4):
        """
        Parameters
        ----------
        channels: Input channel size for all three layers
        scale_factor: Factor to upsample the feature map
        """
        super(InceptionV3Transfer, self).__init__()
        self.scale_factor = scale_factor
        
        
        conv_layers = list(models.inception_v3(pretrained=True).features.children())
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
