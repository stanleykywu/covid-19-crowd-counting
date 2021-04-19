import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch
from data import CrowdClassificationDataSet 
from data import default_train_transform_classification, default_val_transform_classification
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix


def run_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    return parser.parse_args()


def main(args):
    loaders = {
        "train": CrowdClassificationDataSet(
            'part_A/train_data', default_train_transform_classification()
        ),
        "val": CrowdClassificationDataSet(
            'part_A/test_data', default_val_transform_classification()
        ),
        "test": CrowdClassificationDataSet(
            'part_B/test_data', default_val_transform_classification()
        )
    }

    model = torch.load('saved_models/' + args.model)
    model.eval()

    train_vgg16_predictions = []
    train_vgg16_actual = []
    val_vgg16_predictions = []
    val_vgg16_actual = []
    test_vgg16_predictions = []
    test_vgg16_actual = []

    for i, data in enumerate(loaders['train'], 0):
        dt = data

        image = dt['image'].to()
        bin = dt['bin']
        
        model.eval()
        outputs = model(image[None, ...].float())
        expected = torch.Tensor([bin]).type(torch.LongTensor)
        _, preds = torch.max(outputs, 1)
            
        train_vgg16_predictions.append(preds)
        train_vgg16_actual.append(expected)

    for i, data in enumerate(loaders['val'], 0):
        dt = data

        image = dt['image'].to()
        bin = dt['bin']
        
        model.eval()
        outputs = model(image[None, ...].float())
        expected = torch.Tensor([bin]).type(torch.LongTensor)
        _, preds = torch.max(outputs, 1)
            
        val_vgg16_predictions.append(preds)
        val_vgg16_actual.append(expected)

    for i, data in enumerate(loaders['test'], 0):
        dt = data

        image = dt['image'].to()
        bin = dt['bin']
        
        model.eval()
        outputs = model(image[None, ...].float())
        expected = torch.Tensor([bin]).type(torch.LongTensor)
        _, preds = torch.max(outputs, 1)
            
        test_vgg16_predictions.append(preds)
        test_vgg16_actual.append(expected)

    train_acc = accuracy_score(train_vgg16_actual, train_vgg16_predictions)
    val_acc = accuracy_score(val_vgg16_actual, val_vgg16_predictions)
    test_acc = accuracy_score(test_vgg16_actual, test_vgg16_predictions)
    
    print("{}".format(args.model))
    print('================================')
    print('Training Accuracy: {}'.format(train_acc))
    print('Validation Accuracy: {}'.format(val_acc))
    print('Testing Accuracy: {}'.format(test_acc))
    print('================================')

    fg, (p1, p2, p3) = plt.subplots(1, 3, figsize=(15, 4))
    labels = [0, 1, 2, 3, 4]
    cm = confusion_matrix(train_vgg16_actual, train_vgg16_predictions, labels=labels)
    ax = p1.add_subplot(111)
    cax = ax.matshow(cm)
    p1.title('Confusion matrix on Training')
    p1.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    p1.ylabel('Actual')
    p1.xlabel('Predicted')
    
    cm = confusion_matrix(val_vgg16_actual, val_vgg16_predictions, labels=labels)
    ax = p1.add_subplot(111)
    cax = ax.matshow(cm)
    p2.title('Confusion matrix on Validation')
    p2.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    p2.ylabel('Actual')
    p2.xlabel('Predicted')
    
    cm = confusion_matrix(test_vgg16_actual, test_vgg16_predictions, labels=labels)
    ax = p1.add_subplot(111)
    cax = ax.matshow(cm)
    p3.title('Confusion matrix on Testing')
    p3.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    p3.ylabel('Actual')
    p3.xlabel('Predicted')
    
    fg.savefig('results/{}_results'.format(args.model))
    
if __name__=='__main__':
    args = run_argparse()
    main(args)