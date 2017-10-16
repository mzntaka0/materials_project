# -*- coding: utf-8 -*-
"""
search materials which match with the specific bandgap.
"""
import os
import sys
import itertools

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymatgen
import seaborn as sns
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from numpy import zeros, mean
from pymatgen import Composition, Element
from pymatgen import MPRester, periodic_table
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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

def get_feature_vec(materials):
    physicalFeatures = []
    for material in materials:
        theseFeatures = []
        fraction = []
        atomicNo = []
        eneg = []
        group = []

        for element in material:
                fraction.append(material.get_atomic_fraction(element))
                atomicNo.append(float(element.Z))
                eneg.append(element.X)
                group.append(float(element.group))

        mustReverse = False
        if fraction[1] > fraction[0]:
                mustReverse = True

        for features in [fraction, atomicNo, eneg, group]:
                if mustReverse:
                        features.reverse()
        theseFeatures.append(fraction[0] / fraction[1])
        theseFeatures.append(eneg[0] - eneg[1])
        theseFeatures.append(group[0])
        theseFeatures.append(group[1])
        physicalFeatures.append(theseFeatures)
    return np.array(physicalFeatures, dtype=np.float32)


class Net(nn.Module):
    params = dict()
    params['input_size'] = 4 
    params['hidden_size'] = 30 
    params['num_classes'] = 1
    params['num_epochs'] = 5
    params['batch_size'] = 50
    params['learning_rate'] = 0.001
    params['test_size'] = 0.25

    def __init__(self, **kwargs):
        if kwargs:
            for arg_name, value in kwargs.items():
                self.params[arg_name] = value
        super(Net, self).__init__()
        self.batch_norm = nn.BatchNorm1d(50)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(self.params['input_size'], self.params['hidden_size'])
        self.relu = nn.ReLU()
        #self.fc2 = nn.Linear(self.params['hidden_size'], self.params['hidden_size'])
        self.fc2 = nn.Linear(self.params['hidden_size'], self.params['num_classes'])

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    API_KEY = 'tvLpmn5hMTXsVy8G' # You have to register with Materials Project to receive an API
    csv_file_path = 'storage/bandgapDFT.csv'

    df = load_csv(csv_file_path) 


    composition = Composition(df[0][0])
    for element in composition:
        print(composition.get_atomic_fraction(element))


    materials = df.loc[:, 0].values.tolist()
    materials = list(map(lambda m: Composition(m), materials))
    target_band_gap = df.loc[:, 1].values
    y_data = np.array(list(map(lambda t: [t], target_band_gap)), np.float32)
    x_data = get_feature_vec(materials)

    pca = PCA(n_components=2)
    pca.fit(x_data)
    Xd = pca.transform(x_data)
    print(Xd)
    plt.scatter(Xd[:, 0], Xd[:, 1])
    plt.show()

    plt.plot(np.arange(y_data), target_band_gap)
    plt.show()

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
