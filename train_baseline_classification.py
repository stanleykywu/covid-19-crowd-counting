from data import CrowdDataSet, CrowdClassificationDataSet, default_val_transform_classification, default_train_transform_classification
from data import default_train_transforms

from models import BaselineClassification
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
        'part_A/test_data', default_val_transform_classification()
    )
}

model = BaselineClassification()
criterion = nn.CrossEntropyLoss()
lr = 1e-3
momentum = 0.9
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

train_losses, train_accuracies, val_losses, val_accuracies = train_classification(model, loaders, criterion, optimizer, 15)
torch.save(model, 'saved_models/baseline_classification_final')
np.save(f"loss_experiments/baseline_classification_losses", (train_losses, train_accuracies, val_losses, val_accuracies))

