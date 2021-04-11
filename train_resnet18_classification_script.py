from data import CrowdDataSet, CrowdClassificationDataSet, default_test_transforms, default_train_transform_classification
from data import default_train_transforms

from models import ResNet18Classification
from trainer import train, train_classification
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np

loaders = {
    "train": CrowdClassificationDataSet(
        'part_A/train_data', default_train_transform_classification()
    ),
    "val": CrowdClassificationDataSet(
        'part_A/test_data', default_train_transform_classification()
    )
}

model = ResNet18Classification()
criterion = nn.CrossEntropyLoss()
lr = 1e-3
momentum = 0.9
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

losses = train_classification(model, loaders['train'], criterion, optimizer, 700)
torch.save(model, 'saved_models/vgg16_classification')
np.save(f"loss_experiments/vgg16_classification_losses", (losses))

