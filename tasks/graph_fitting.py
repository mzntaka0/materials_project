# -*- coding: utf-8 -*-
"""
"""
import os
import sys

import pandas as pd
import numpy  as np
from bpdb import set_trace
import matplotlib.pyplot as plt
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.nn.modules.distance import CosineSimilarity
from torchvision import transforms
from tqdm import tqdm


def load_csv(csv_file_path):
    return pd.read_csv(
            csv_file_path,
            header=None,
            )

class Net(nn.Module):

    def __init__(self, **kwargs):
        self.params = dict()
        self.params['input_size'] = 1
        self.params['hidden_size'] = 20
        self.params['num_classes'] = 1
        self.params['num_epochs'] = 100
        self.params['batch_size'] = 50
        self.params['learning_rate'] = 0.001
        self.params['test_size'] = 0.25

        if kwargs:
            for arg_name, value in kwargs.items():
                self.params[arg_name] = value
        super(Net, self).__init__()
        self.fc1 = nn.Linear(self.params['input_size'], self.params['hidden_size'])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(self.params['hidden_size'], self.params['hidden_size'])
        self.fc2 = nn.Linear(self.params['hidden_size'], self.params['hidden_size'])
        self.fc3 = nn.Linear(self.params['hidden_size'], self.params['num_classes'])


    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    #csv_file_path = 'storage/vl1924e_1.csv'
    csv_file_path = 'storage/bandgapDFT.csv'
    df = load_csv(csv_file_path)
    x_data = df.loc[:, 0].values.reshape(-1, 1)
    y_data = df.loc[:, 1].values.reshape(-1, 1)

    x_data = x_data.astype(np.float32)
    y_data = y_data.astype(np.float32)

    net = Net()
    x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data,
            test_size=net.params['test_size']
            )

    criterion = nn.MSELoss(size_average=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=net.params['learning_rate'])

    N = len(y_train)
    batch_size = net.params['batch_size']

    for epoch in range(net.params['num_epochs']):
        perm = np.random.permutation(N)
        net.train()
        for i, n in enumerate(range(0, N, net.params['batch_size'])):
            x_batch = x_train[perm[n:n+net.params['batch_size']]]
            y_batch = y_train[perm[n:n+net.params['batch_size']]]
            x_batch = Variable(torch.from_numpy(x_batch))
            y_batch = Variable(torch.from_numpy(y_batch))
            optimizer.zero_grad()  
            outputs = net(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            if (i+1) % 2 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                    %(epoch+1, net.params['num_epochs'], i+1, len(x_train)//batch_size, loss.data[0]))

    print(
            """
            ###
            training done
            ###
            """
            )

    correct = 0
    total = 0
    N_test = len(y_test)
    print(N_test)
    for n in range(0, N_test, net.params['batch_size']):
        x_batch = x_train[perm[n:n+net.params['batch_size']]]
        y_batch = y_train[perm[n:n+net.params['batch_size']]]
        x_batch = Variable(torch.from_numpy(x_batch))
        y_batch = Variable(torch.from_numpy(y_batch))
        outputs = net(x_batch)
        print(criterion(outputs, y_batch))

    x_test = Variable(torch.from_numpy(np.arange(361, 820, 0.02, dtype=np.float32).reshape(-1, 1)))
    print(x_test)
    outputs = net(x_test)

    print(x_test)
    print(outputs)

    plt.scatter(x_test.data.numpy(), outputs.data.numpy())
    plt.show()

