import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable


class BinaryLogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x


if __name__ == '__main__':
    torch.manual_seed(0)

    '''
    Load data
    '''
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 1], dtype=np.float32).reshape(-1, 1)

    '''
    Build model
    '''
    model = BinaryLogisticRegression(input_dim=2)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    '''
    Train model
    '''
    train_loss = []
    model.train()
    epochs = 300
    for epoch in range(epochs):
        data = Variable(torch.from_numpy(X))
        labels = Variable(torch.from_numpy(y))

        optimizer.zero_grad()
        preds = model(data)
        loss = criterion(preds, labels)

        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    '''
    Evaluate model
    '''
    val_loss = []
    model.eval()
    data = Variable(torch.from_numpy(X))
    labels = Variable(torch.from_numpy(y))

    preds = model(data)
    loss = criterion(preds, labels)
    val_loss.append(loss)

    predicted = (preds.data > 0.5).float()
    correct = torch.eq(predicted, labels).sum()
    acc = correct / len(X)
    print('acc: {:.3}'.format(acc.float().item()))
