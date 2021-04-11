from data import CrowdClassificationDataSet 
import matplotlib.pyplot as plt
import numpy as np
from data import CrowdDataSet 
from data import default_train_transform_classification
import torch

loaders = {
    "train": CrowdClassificationDataSet(
        'part_A/train_data', default_train_transform_classification()
    ),
    "val": CrowdClassificationDataSet(
        'part_A/test_data', default_train_transform_classification()
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
    predictions = np.array([x for x in predictions])
    print(predictions, bin)
    print('prediction: {}'.format(np.argmax(predictions)))
        
    train_vgg16_predictions.append(np.argmax(predictions))
    train_vgg16_actual.append(bin)