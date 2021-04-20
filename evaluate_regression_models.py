import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch
from data import CrowdDataSet 
from data import default_train_transforms, default_val_transforms
import argparse
from utils import get_density_map_gaussian
from sklearn.metrics import mean_squared_error, r2_score 
import matplotlib.gridspec as gridspes


def run_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    return parser.parse_args()


def main(args):
    loaders = {
        "train": CrowdDataSet(
            'part_A/train_data', default_train_transforms()
        ),
        "val": CrowdDataSet(
            'part_A/test_data', default_val_transforms()
        ),
        "test_unbalanced": CrowdDataSet(
            'part_B/test_data', default_val_transforms()
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
        gt = dt['gt']
        
        model.eval()
        predictions = model(image[None, ...].float())
        predictions = predictions.squeeze().data.cpu().numpy() 
        count = np.sum(predictions) / 100
            
        train_vgg16_predictions.append(count)
        train_vgg16_actual.append(len(gt))

    limit = int(len(loaders['val']) / 2)
    print('Evaluating Validation...')
    for i, data in enumerate(loaders['val'], 0):
        if i >= limit: break
        dt = data

        image = dt['image'].to()
        gt = dt['gt']
        
        model.eval()
        predictions = model(image[None, ...].float())
        predictions = predictions.squeeze().data.cpu().numpy() 
        count = np.sum(predictions) / 100
            
        val_vgg16_predictions.append(count)
        val_vgg16_actual.append(len(gt))

    print('Evaluating Testing (Balanced)...')
    for i, data in enumerate(loaders['val'], limit):
        dt = data

        image = dt['image'].to()
        gt = dt['gt']
        
        model.eval()
        predictions = model(image[None, ...].float())
        predictions = predictions.squeeze().data.cpu().numpy() 
        count = np.sum(predictions) / 100
            
        test_b_vgg16_predictions.append(count)
        test_b_vgg16_actual.append(len(gt))

    print('Evaluating Testing (Unbalanced)...')
    for i, data in enumerate(loaders['test_unbalanced'], 0):
        dt = data

        image = dt['image'].to()
        gt = dt['gt']
        
        model.eval()
        predictions = model(image[None, ...].float())
        predictions = predictions.squeeze().data.cpu().numpy() 
        count = np.sum(predictions) / 100
            
        test_ub_vgg16_predictions.append(count)
        test_ub_vgg16_actual.append(len(gt))

    
    train_r2 = r2_score(train_vgg16_actual, train_vgg16_predictions)
    val_r2 = r2_score(val_vgg16_actual, val_vgg16_predictions)
    test_b_r2 = r2_score(test_b_vgg16_actual, test_b_vgg16_predictions)
    test_ub_r2 = r2_score(test_ub_vgg16_actual, test_ub_vgg16_predictions)
    train_mse = mean_squared_error(train_vgg16_actual, train_vgg16_predictions)
    val_mse = mean_squared_error(val_vgg16_actual, val_vgg16_predictions)
    test_b_mse = mean_squared_error(test_b_vgg16_actual, test_b_vgg16_predictions)
    test_ub_mse = mean_squared_error(test_ub_vgg16_actual, test_ub_vgg16_predictions)

    print("{}".format(args.model))
    print('================================')
    print('Training r2: {}'.format(train_r2))
    print('Validation r2: {}'.format(val_r2))
    print('Testing (BalanceD) r2: {}'.format(test_b_r2))
    print('Testing (Unbalanced) r2: {}'.format(test_ub_r2))
    print('================================')
    print('Training MSE: {}'.format(train_mse))
    print('Validation MSE: {}'.format(val_mse))
    print('Testing (Balanced) MSE: {}'.format(test_b_mse))
    print('Testing (Unbalanced) MSE: {}'.format(test_ub_mse))

    fg, (p1, p2, p3, p4) = plt.subplots(1, 4, figsize=(15, 4))
    x = np.linspace(0, max(train_vgg16_actual), 1000)
    y = x
    p1.plot(x, y, '-r', label='Ground Truths')
    p1.scatter(train_vgg16_actual, train_vgg16_predictions, label='Training Data')
    p1.legend()
    p1.set_title('Training\nMSE: {:.2f}\n r2: {:.2f}'.format(train_mse, train_r2))

    x = np.linspace(0, max(val_vgg16_actual), 1000)
    y = x
    p2.plot(x, y, '-r', label='Ground Truths')
    p2.scatter(val_vgg16_actual, val_vgg16_predictions, label='Validation Data')
    p2.legend()
    p2.set_title('Validation\nMSE: {:.2f}\n r2: {:.2f}'.format(val_mse, val_r2))

    x = np.linspace(0, max(test_b_vgg16_actual), 1000)
    y = x
    p3.plot(x, y, '-r', label='Ground Truths')
    p3.scatter(test_b_vgg16_actual, test_b_vgg16_predictions, label='Testing (Balanced) Data')
    p3.legend()
    p3.set_title('Testing (Balanced)\nMSE: {:.2f}\n r2: {:.2f}'.format(test_b_mse, test_b_r2))

    x = np.linspace(0, max(test_ub_vgg16_actual), 1000)
    y = x
    p4.plot(x, y, '-r', label='Ground Truths')
    p4.scatter(test_ub_vgg16_actual, test_ub_vgg16_predictions, label='Testing (Unbalanced) Data')
    p4.legend()
    p4.set_title('Testing (Unbalanced)\nMSE: {:.2f}\n r2: {:.2f}'.format(test_ub_mse, test_ub_r2))
    fg.text(0.04, 0.5, 'Predicted', va='center', rotation='vertical')
    fg.text(0.5, 0.04, 'Actual', ha='center')

    fg.savefig('results/{}_results'.format(args.model))

    
if __name__=='__main__':
    args = run_argparse()
    main(args)