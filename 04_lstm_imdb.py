import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from keras.datasets.imdb import load_data
from utils import pad_sequences, sort_sequences


class RNN(nn.Module):
    '''
    IMDb classification with stacked LSTM
    '''
    def __init__(self, vocab):
        super().__init__()

        self._lstm = nn.Sequential(
            nn.Embedding(vocab, 100),
            nn.LSTM(100, 50, num_layers=2, batch_first=True),
        )
        self._lienar = nn.Sequential(
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x, _ = self._lstm(x)
        x = x[:, -1]
        x = self._lienar(x)
        return x


if __name__ == '__main__':
    torch.manual_seed(0)
    use_gpu = torch.cuda.is_available()

    '''
    Load data
    '''
    pad_value = 0
    batch_size = 100
    num_words = 10000
    (train_x, train_y), (test_x, test_y) = load_data(num_words=num_words)

    train_x, train_y = sort_sequences(train_x, train_y[:, np.newaxis])
    test_x, test_y = sort_sequences(test_x, test_y[:, np.newaxis])

    train_x, train_y = train_x[:10000], train_y[:10000]
    test_x, test_y = test_x[:2000], test_y[:2000]

    '''
    Build model
    '''
    model = RNN(num_words)
    if use_gpu:
        model.cuda()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    '''
    Train model
    '''
    train_loss = []
    model.train()
    epochs = 20
    n_batches = len(train_x) // batch_size

    for epoch in range(epochs):
        cost = 0.
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size

            data = pad_sequences(train_x[start:end], value=pad_value)
            labels = np.array(train_y[start:end])
            data = torch.from_numpy(data).long()
            labels = torch.from_numpy(labels).float()

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
        cost /= n_batches
        train_loss.append(cost)
        print('epochs: {}, loss: {:.3}'.format(epoch+1, cost))

    '''
    Evaluate model
    '''
    val_loss = []
    model.eval()
    n_batches = len(test_x) // batch_size
    cost = 0.
    correct = 0
    total = 0
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        data = pad_sequences(test_x[start:end], value=pad_value)
        labels = np.array(test_y[start:end])
        data = torch.from_numpy(data).long()
        labels = torch.from_numpy(labels).float()

        if use_gpu:
            data = Variable(data.cuda())
            labels = Variable(labels.cuda())
        else:
            data = Variable(data)
            labels = Variable(labels)

        preds = model(data)
        loss = criterion(preds, labels)
        cost += loss.item()
        predicted = (preds.data > 0.5).float()
        correct += torch.eq(predicted, labels).sum()
        total += labels.size(0)

    cost /= n_batches
    acc = correct.float() / total
    val_loss.append(cost)
    print('val_loss: {:.3}, val_acc: {:.3}'.format(cost, acc.item()))
