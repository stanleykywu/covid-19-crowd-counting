from data import CrowdDataSet 
from data import default_train_transforms, default_test_transforms 
import numpy as np
from models import ResNetTransfer
from trainer import train
import torch.optim as optim
import torch.nn as nn
import torch

loaders = {
    "train": CrowdDataSet(
        'part_A/train_data', default_train_transforms()
    ),
    "val": CrowdDataSet(
        'part_A/test_data', default_test_transforms()
    )
}

model = ResNetTransfer(scale_factor=32) 
criterion = nn.MSELoss()
lr = 1e-5
optimizer = optim.Adam(model.parameters(), lr=lr)

losses, r2 = train(model, loaders['train'], criterion, optimizer, 700)
torch.save(model, 'saved_models/resnet18_adaptive')
np.save(f"loss_experiments/resnet18_adaptive_losses", (losses, r2))
