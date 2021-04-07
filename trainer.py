import torch
import numpy as np

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

def trainInception(model, trainloader, criterion, optimizer, epochs):
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        for i in range(0, len(trainloader), 5):
            images = torch.Tensor([trainloader[i]['image'].numpy() for i in range(i, i + 5)])
            densities = torch.Tensor([trainloader[i]['den'] for i in range(i, i + 5)])

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, densities)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 0:    # print every 5 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0