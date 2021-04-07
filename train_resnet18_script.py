from data import CrowdDataSet 
from data import default_train_transforms
from torchvision import transforms
from PIL import Image
from utils import get_density_map_gaussian
import numpy as np

from models import VGG16Transfer, ResNetTransfer, InceptionV3Transfer
from trainer import train, trainInception
import torch.optim as optim
import torch.nn as nn
import torch

loaders = {
    "train": CrowdDataSet(
        'part_A/train_data', default_train_transforms(output_size=224, factor=1)
    ),
    "val": CrowdDataSet(
        'part_A/test_data', default_train_transforms(output_size=224, factor=1)
    )
}

model = ResNetTransfer(scale_factor=32) 
criterion = nn.MSELoss()
lr = 1e-5
optimizer = optim.Adam(model.parameters(), lr=lr)

losses = train(model, loaders['train'], criterion, optimizer, 500)
torch.save(model, 'saved_models/resnet18_only_crop')
np.save(f"loss_experiments/resnet18_losses", (losses))
