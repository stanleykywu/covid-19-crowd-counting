import torch
import numpy as np
from sklearn.metrics import r2_score

def train_classification(model, trainloader, criterion, optimizer, epochs):
    losses = []
    accuracies = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_correct = 0
        for i, data in enumerate(trainloader, 0):
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
            running_loss += loss.item()
            running_correct += torch.sum(preds == bin)

        losses.append(running_loss)
        accuracies.append(running_correct / len(trainloader))
        print('Epoch: {}, loss: {}, training acc: {}'.format(epoch, losses[-1], accuracies[-1]))
        
    return losses, accuracies

def train(model, trainloader, criterion, optimizer, epochs):
    losses = []
    r2 = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_expected = []
        running_predicted = []
        for i, data in enumerate(trainloader, 0):
            model.train()
            # get the inputs; data is a list of [inputs, labels]
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
            running_loss += loss.item()
            prediction = outputs.squeeze().data.cpu().numpy() 
            count = np.sum(prediction) / 100
            running_predicted.append(count)
            running_expected.append(len(dt['gt']))

        losses.append(running_loss)
        r2.append(r2_score(running_expected, running_predicted))
        print('Epoch: {}, loss: {}, r2: {}'.format(epoch, losses[-1], r2[-1]))

    return losses, r2