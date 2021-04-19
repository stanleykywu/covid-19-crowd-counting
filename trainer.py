import torch
import numpy as np
from sklearn.metrics import r2_score

def train_classification(model, loader, criterion, optimizer, epochs):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(epochs): 
        train_running_loss = 0
        train_running_correct = 0
        val_running_loss = 0
        val_running_correct = 0

        for data in loader['train']:
            model.train()
            dt = data

            image = dt['image'].to()
            bin = dt['bin']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(image[None, ...].float())
            expected = torch.Tensor([bin]).type(torch.LongTensor)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, expected)
            loss.backward()
            optimizer.step()

            # print statistics
            train_running_loss += loss.item()
            train_running_correct += torch.sum(preds == bin)

        if len(train_losses) > 1 and abs(train_running_loss - train_losses[-1]) <= 0.1:
            # early stopping
            return train_losses, train_accuracies, val_losses, val_accuracies

        train_losses.append(train_running_loss)
        train_accuracies.append(train_running_correct / len(loader['train']))

        limit = len(loader['val']) / 2
        for i, data in enumerate(loader['val'], 0):
            if i >= limit: break
            model.eval()
            dt = data

            image = dt['image'].to()
            bin = dt['bin']

            outputs = model(image[None, ...].float())
            expected = torch.Tensor([bin]).type(torch.LongTensor)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, expected)

            # print statistics
            val_running_loss += loss.item()
            val_running_correct += torch.sum(preds == bin)
    
        val_losses.append(val_running_loss)
        val_accuracies.append(val_running_correct / len(loader['val']))

        print('Epoch: {}, training loss: {}, training acc: {}'.format(epoch, train_losses[-1], train_accuracies[-1]))
        print('\t validation loss: {}, validation acc: {}'.format(val_losses[-1], val_accuracies[-1]))
        
    return train_losses, train_accuracies, val_losses, val_accuracies

def train(model, loader, criterion, optimizer, epochs):
    train_losses = []
    train_r2 = []
    val_losses = []
    val_r2 = []

    for epoch in range(epochs): 
        train_running_loss = 0
        train_running_expected = []
        train_running_predicted = []
        val_running_loss = 0
        val_running_expected = []
        val_running_predicted = []

        for data in loader['train']:
            model.train()
            dt = data

            image = dt['image'].to()
            den = dt['den']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(image[None, ...].float())
            loss = criterion(outputs, torch.from_numpy(den[None, ...]))
            loss.backward()
            optimizer.step()

            # print statistics
            train_running_loss += loss.item()
            prediction = outputs.squeeze().data.cpu().numpy() 
            count = np.sum(prediction) / 100
            train_running_predicted.append(count)
            train_running_expected.append(len(dt['gt']))

        if len(train_losses) > 1 and abs(train_running_loss - train_losses[-1]) <= 0.1:    
            # early stopping
            return train_losses, train_r2, val_losses, val_r2


        train_losses.append(train_running_loss)
        train_r2.append(r2_score(train_running_expected, train_running_predicted))

        limit = len(loader['val']) / 2
        for i, data in enumerate(loader['val'], 0):
            if i >= limit: break
            model.eval()
            dt = data

            image = dt['image'].to()
            den = dt['den']

            outputs = model(image[None, ...].float())
            loss = criterion(outputs, torch.from_numpy(den[None, ...]))

            # print statistics
            val_running_loss += loss.item()
            prediction = outputs.squeeze().data.cpu().numpy() 
            count = np.sum(prediction) / 100
            val_running_predicted.append(count)
            val_running_expected.append(len(dt['gt']))
    
        val_losses.append(val_running_loss)
        val_r2.append(r2_score(val_running_expected, val_running_predicted))

        print('Epoch: {}, training loss: {}, training r2: {}'.format(epoch, train_losses[-1], train_r2[-1]))
        print('\t validation loss: {}, validation r2: {}'.format(val_losses[-1], val_r2[-1]))

    return train_losses, train_r2, val_losses, val_r2