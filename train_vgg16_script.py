from data import CrowdDataSet 
from data import default_train_transforms, default_val_transforms 
import numpy as np
from models import VGG16Transfer
from trainer import train
import torch.optim as optim
import torch.nn as nn
import torch

loaders = {
    "train": CrowdDataSet(
        'part_A/train_data', default_train_transforms()
    ),
    "val": CrowdDataSet(
        'part_A/test_data', default_val_transforms()
    )
}

model = VGG16Transfer() 
criterion = nn.MSELoss()
lr = 1e-5
optimizer = optim.Adam(model.parameters(), lr=lr)

train_losses, train_r2, val_losses, val_r2 = train(model, loaders, criterion, optimizer, 20)
torch.save(model, 'saved_models/vgg16_density_map')
np.save(f"loss_experiments/vgg16_losses_adaptive_final", (train_losses, train_r2, val_losses, val_r2))

