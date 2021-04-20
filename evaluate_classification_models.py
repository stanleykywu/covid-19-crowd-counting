import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch
from data import CrowdClassificationDataSet 
from data import default_train_transform_classification, default_val_transform_classification
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score


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
        "test_unbalanced": CrowdClassificationDataSet(
            'part_B/test_data', default_val_transform_classification()
        )
    }

    model = torch.load('saved_models/' + args.model)
    model.eval()

    train_vgg16_predictions = []
    train_vgg16_actual = []
    val_vgg16_predictions = []
    val_vgg16_actual = []
    test_b_vgg16_predictions = []
    test_b_vgg16_actual = []
    test_ub_vgg16_predictions = []
    test_ub_vgg16_actual = []

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

    limit = int(len(loaders['val']) / 2)
    print('Evaluating Validation...')
    for i, data in enumerate(loaders['val'], 0):
        if i >= limit: break
        dt = data

        image = dt['image'].to()
        bin = dt['bin']
        
        model.eval()
        outputs = model(image[None, ...].float())
        expected = torch.Tensor([bin]).type(torch.LongTensor)
        _, preds = torch.max(outputs, 1)
            
        val_vgg16_predictions.append(preds)
        val_vgg16_actual.append(expected)

    print('Evaluating Testing Balanced...')
    for i, data in enumerate(loaders['val'], limit):
        dt = data

        image = dt['image'].to()
        bin = dt['bin']
        
        model.eval()
        outputs = model(image[None, ...].float())
        expected = torch.Tensor([bin]).type(torch.LongTensor)
        _, preds = torch.max(outputs, 1)
            
        test_b_vgg16_predictions.append(preds)
        test_b_vgg16_actual.append(expected)

    print('Evaluating Testing Unbalanced...')
    for i, data in enumerate(loaders['test_unbalanced'], 0):
        dt = data

        image = dt['image'].to()
        bin = dt['bin']
        
        model.eval()
        outputs = model(image[None, ...].float())
        expected = torch.Tensor([bin]).type(torch.LongTensor)
        _, preds = torch.max(outputs, 1)
            
        test_ub_vgg16_predictions.append(preds)
        test_ub_vgg16_actual.append(expected)

    train_acc = accuracy_score(train_vgg16_actual, train_vgg16_predictions)
    val_acc = accuracy_score(val_vgg16_actual, val_vgg16_predictions)
    test_b_acc = accuracy_score(test_b_vgg16_actual, test_b_vgg16_predictions)
    test_ub_acc = accuracy_score(test_ub_vgg16_actual, test_ub_vgg16_predictions)

    train_f1 = f1_score(train_vgg16_actual, train_vgg16_predictions, average='weighted')
    val_f1 = f1_score(val_vgg16_actual, val_vgg16_predictions, average='weighted')
    test_b_f1 = f1_score(test_b_vgg16_actual, test_b_vgg16_predictions, average='weighted')
    test_ub_f1 = f1_score(test_ub_vgg16_actual, test_ub_vgg16_predictions, average='weighted')

    train_prec = precision_score(train_vgg16_actual, train_vgg16_predictions, average='weighted')
    val_prec = precision_score(val_vgg16_actual, val_vgg16_predictions, average='weighted')
    test_b_prec = precision_score(test_b_vgg16_actual, test_b_vgg16_predictions, average='weighted')
    test_ub_prec = precision_score(test_ub_vgg16_actual, test_ub_vgg16_predictions, average='weighted')

    train_recall = recall_score(train_vgg16_actual, train_vgg16_predictions, average='weighted')
    val_recall = recall_score(val_vgg16_actual, val_vgg16_predictions, average='weighted')
    test_b_recall = recall_score(test_b_vgg16_actual, test_b_vgg16_predictions, average='weighted')
    test_ub_recall = recall_score(test_ub_vgg16_actual, test_ub_vgg16_predictions, average='weighted')
    
    print("{}".format(args.model))
    print('================================')
    print('Training Accuracy: {}'.format(train_acc))
    print('Validation Accuracy: {}'.format(val_acc))
    print('Testing (Balanced) Accuracy: {}'.format(test_b_acc))
    print('Testing (Unbalanced) Accuracy: {}'.format(test_ub_acc))
    print('================================')
    print('Training f1: {}'.format(train_acc))
    print('Validation f1: {}'.format(val_acc))
    print('Testing (Balanced) f1: {}'.format(test_b_acc))
    print('Testing (Unbalanced) f1: {}'.format(test_ub_acc))
    print('================================')
    print('Training Precision: {}'.format(train_prec))
    print('Validation Precision: {}'.format(val_prec))
    print('Testing (Balanced) Precision: {}'.format(test_b_prec))
    print('Testing (Unbalanced) Precision: {}'.format(test_ub_prec))
    print('================================')
    print('Training Recall: {}'.format(train_recall))
    print('Validation Recall: {}'.format(val_recall))
    print('Testing (Balanced) Recall: {}'.format(test_b_recall))
    print('Testing (Unbalanced) Recall: {}'.format(test_ub_recall))
    print('================================')

    fg, (p1, p2, p3, p4) = plt.subplots(1, 4, figsize=(15, 4))
    cf_matrix = confusion_matrix(train_vgg16_actual, train_vgg16_predictions, labels=[0, 1, 2, 3, 4])
    disp = ConfusionMatrixDisplay(cf_matrix, display_labels=[0, 1, 2, 3, 4])
    disp.plot(ax=p1)
    disp.ax_.set_title('Training Acc: {:.2f}\n f1: {:.2f}'.format(train_acc, train_f1))
    disp.im_.colorbar.remove()
    disp.ax_.set_xlabel('')

    cf_matrix = confusion_matrix(val_vgg16_actual, val_vgg16_predictions, labels=[0, 1, 2, 3, 4])
    disp = ConfusionMatrixDisplay(cf_matrix, display_labels=[0, 1, 2, 3, 4])
    disp.plot(ax=p2)
    disp.ax_.set_title('Validation Acc: {:.2f}\n f1: {:.2f}'.format(val_acc, val_f1))
    disp.im_.colorbar.remove()
    disp.ax_.set_xlabel('')
    disp.ax_.set_ylabel('')

    cf_matrix = confusion_matrix(test_b_vgg16_actual, test_b_vgg16_predictions, labels=[0, 1, 2, 3, 4])
    disp = ConfusionMatrixDisplay(cf_matrix, display_labels=[0, 1, 2, 3, 4])
    disp.plot(ax=p3)
    disp.ax_.set_title('Testing (Balanced) Acc: {:.2f}\n f1: {:.2f}'.format(test_b_acc, test_b_f1))
    disp.im_.colorbar.remove()
    disp.ax_.set_xlabel('')
    disp.ax_.set_ylabel('')

    cf_matrix = confusion_matrix(test_ub_vgg16_actual, test_ub_vgg16_predictions, labels=[0, 1, 2, 3, 4])
    disp = ConfusionMatrixDisplay(cf_matrix, display_labels=[0, 1, 2, 3, 4])
    disp.plot(ax=p4)
    disp.ax_.set_title('Testing (Unbalanced) Acc: {:.2f}\n f1: {:.2f}'.format(test_ub_acc, test_ub_f1))
    disp.im_.colorbar.remove()
    disp.ax_.set_xlabel('')
    disp.ax_.set_ylabel('')

    fg.text(0.04, 0.5, 'Actual Label', va='center', rotation='vertical')
    fg.text(0.5, 0.04, 'Predicted Label', ha='center')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)

    fg.colorbar(disp.im_, ax=(p1, p2, p3, p4))
    fg.savefig('results/{}_results'.format(args.model))
    
if __name__=='__main__':
    args = run_argparse()
    main(args)