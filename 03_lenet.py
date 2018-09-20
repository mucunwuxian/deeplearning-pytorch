import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from keras.datasets.mnist import load_data


class CNN(nn.Module):
    '''
    MNIST classification with LeNet-based CNN
    '''
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x


if __name__ == '__main__':
    torch.manual_seed(0)
    use_gpu = torch.cuda.is_available()

    '''
    Load data
    '''
    batch_size = 100
    (train_X, train_y), (test_X, test_y) = load_data()

    train_X = torch.from_numpy(train_X).float().unsqueeze(-3)
    train_y = torch.from_numpy(train_y).long()
    test_X = torch.from_numpy(test_X).float().unsqueeze(-3)
    test_y = torch.from_numpy(test_y).long()

    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    '''
    Build model
    '''
    model = CNN()
    print(model)
    if use_gpu:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    '''
    Train model
    '''
    train_loss = []
    model.train()
    epochs = 5
    for epoch in range(epochs):
        cost = 0.
        for batch_idx, (data, labels) in enumerate(train_loader):
            if use_gpu:
                data = Variable(data.cuda())
                labels = Variable(labels.cuda())
            else:
                data = Variable(data)
                labels = Variable(labels)

            optimizer.zero_grad()
            preds = model(data)
            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()
            cost += loss.item()
        cost /= len(train_loader)
        train_loss.append(cost)
        print('epochs: {}, loss: {:.3}'.format(epoch+1, cost))

    '''
    Evaluate model
    '''
    val_loss = []
    model.eval()
    cost = 0.
    correct = 0
    total = 0
    for batch_idx, (data, labels) in enumerate(test_loader):
        if use_gpu:
            data = Variable(data.cuda())
            labels = Variable(labels.cuda())
        else:
            data = Variable(data)
            labels = Variable(labels)

        preds = model(data)
        loss = criterion(preds, labels)
        cost += loss.item()
        _, predicted = torch.max(preds.data, 1)
        correct += (predicted == labels.data).sum()
        total += labels.size(0)

    cost /= len(test_loader)
    acc = correct.float() / total
    val_loss.append(cost)
    print('val_loss: {:.3}, val_acc: {:.3}'.format(cost, acc.item()))
