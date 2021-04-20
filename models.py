import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGG16Transfer(nn.Module):
    def __init__(self):
        super(VGG16Transfer, self).__init__()
        conv_layers = list(models.vgg16(pretrained=True).features.children())
        
        for layer in conv_layers:
            layer.requires_grad = False

        self.model = nn.Sequential(
            *conv_layers,
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    
    def forward(self, inputs):
        return F.upsample(self.model(inputs), scale_factor=32)


class ResNetTransfer(nn.Module):
    def __init__(self):
        super(ResNetTransfer, self).__init__()
        
        conv_layers = list(models.resnet18(pretrained=True).children())[:8]
        for layer in conv_layers:
            layer.requires_grad = True

        self.model = nn.Sequential(
            # *conv_layers,
        #     nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 128, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 1, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True)
        # )
            *conv_layers,
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    
    def forward(self, inputs):
        return F.upsample(self.model(inputs), scale_factor=32)

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

class BaselineClassification(nn.Module):
    def __init__(self, bins=5):
        super(BaselineClassification, self).__init__()
        self.bins = bins

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(6, 3, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=9075, out_features=4000, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=4000, out_features=2000, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=2000, out_features=98, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=98, out_features=self.bins, bias=True),
        )
    
    def forward(self, inputs):
        x = self.cnn_layers(inputs)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
