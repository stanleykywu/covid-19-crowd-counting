import torch
import numpy as np

def train_classification(model, trainloader, criterion, optimizer, epochs):
    losses = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
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
            loss = criterion(outputs, expected)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        losses.append(running_loss)
        running_loss = 0.0
    return losses

def train(model, trainloader, criterion, optimizer, epochs):
    losses = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
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
        losses.append(running_loss)
        running_loss = 0.0
    return losses