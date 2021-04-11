from data import CrowdDataSet, default_test_transforms 
from data import default_train_transforms
from models import VGG16Transfer
from trainer import train
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np

loaders = {
    "train": CrowdDataSet(
        'part_A/train_data', default_train_transforms()
    ),
    "val": CrowdDataSet(
        'part_A/test_data', default_test_transforms()
    )
}

model = VGG16Transfer(scale_factor=32) 
criterion = nn.MSELoss()
lr = 1e-5
optimizer = optim.Adam(model.parameters(), lr=lr)

losses, r2 = train(model, loaders['train'], criterion, optimizer, 700)
torch.save(model, 'saved_models/vgg16_adaptive')
np.save(f"loss_experiments/vgg16_losses_adaptive", (losses, r2))

