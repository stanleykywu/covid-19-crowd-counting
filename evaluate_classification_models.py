import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch
from data import CrowdClassificationDataSet 
from data import default_train_transform_classification, default_val_transform_classification
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


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

    print('Evaluating Training...')
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

    print('Evaluating Validation...')
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

    print('Evaluating Testing...')
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
    cf_matrix = confusion_matrix(train_vgg16_actual, train_vgg16_predictions)
    disp = ConfusionMatrixDisplay(cf_matrix, display_labels=[0, 1, 2, 3, 4])
    disp.plot(ax=p1)
    disp.ax_.set_title('Training')
    disp.im_.colorbar.remove()
    disp.ax_.set_xlabel('')

    cf_matrix = confusion_matrix(val_vgg16_actual, val_vgg16_predictions)
    disp = ConfusionMatrixDisplay(cf_matrix, display_labels=[0, 1, 2, 3, 4])
    disp.plot(ax=p2)
    disp.ax_.set_title('Validation')
    disp.im_.colorbar.remove()
    disp.ax_.set_xlabel('')
    disp.ax_.set_ylabel('')

    cf_matrix = confusion_matrix(test_vgg16_actual, test_vgg16_predictions)
    disp = ConfusionMatrixDisplay(cf_matrix, display_labels=[0, 1, 2, 3, 4])
    disp.plot(ax=p3)
    disp.ax_.set_title('Testing')
    disp.im_.colorbar.remove()
    disp.ax_.set_xlabel('')
    disp.ax_.set_ylabel('')

    fg.text(0.4, 0.1, 'Predicted label', ha='left')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)


    fg.colorbar(disp.im_, ax=(p1, p2, p3))
    fg.savefig('results/{}_results'.format(args.model))
    
if __name__=='__main__':
    args = run_argparse()
    main(args)