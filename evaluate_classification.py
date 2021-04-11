from data import CrowdDataSet 
import matplotlib.pyplot as plt
import numpy as np
from data import CrowdDataSet 
from data import default_train_transforms, default_test_transforms
import torch

loaders = {
    "train": CrowdDataSet(
        'part_A/train_data', default_train_transforms()
    ),
    "val": CrowdDataSet(
        'part_A/test_data', default_test_transforms()
    )
}

model = torch.load('saved_models/vgg16_classification')
model.eval()

train_vgg16_predictions = []
train_vgg16_actual = []

for i, data in enumerate(loaders['train'], 0):
    dt = data

    image = dt['image'].to()
    bin = dt['bin']
    
    model.eval()
    predictions = model(image[None, ...].float())
    print(predictions, bin)
    print(np.argmax(predictions))
        
    train_vgg16_predictions.append(0)
    train_vgg16_actual.append(0)