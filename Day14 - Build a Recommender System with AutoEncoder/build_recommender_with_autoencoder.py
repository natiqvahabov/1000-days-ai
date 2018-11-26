#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:15:09 2018

@author: natig
"""

# import libraries and pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# importing dataset 1million movies
movies  = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users   = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# prepare train & test set and convert them into INT64 array
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t') 
test_set     = pd.read_csv('ml-100k/u1.test', delimiter='\t')

training_set = np.array(training_set, dtype='int')
test_set     = np.array(test_set, dtype='int')

# matrice -> row- users, column- movies, cell- ratings
# 2 matrices, one for training another for test
# if user u didn't reviewed movie i , cell become 0

# getting number of users and movies
total_number_users  = int(max(max(training_set[:,0]), max(test_set[:, 0])))
total_number_movies = int(max(max(training_set[:,1]), max(test_set[:, 1])))

# function for converting dataset to list of list. because torch needs this type
def convert(data):
    new_data = []
    for id_users in range(1, total_number_users+1):
        id_movies  = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(total_number_movies)
        ratings[id_movies-1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# converting data to Torch tensors
training_set = torch.FloatTensor(training_set)
test_set     = torch.FloatTensor(test_set)

# inheritence from NN module of PyTorch
class SAE(nn.Module):
    def __init__(self,):
        super(SAE, self).__init__()
        # first full connection
        self.fc1 = nn.Linear(total_number_movies, 20)
        # second full connection
        self.fc2 = nn.Linear(20, 10)
        # third full connection between 2nd and 3rd hidden layers
        self.fc3 = nn.Linear(10, 20)
        # fourth full connection
        self.fc4 = nn.Linear(20, total_number_movies)
        # adding sigmoid activation function
        self.activation = nn.Sigmoid()
    def forward(self, x):
        # encoding
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        # decoding
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
# creating object of SAE class
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

# training SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(total_number_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data>0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = total_number_movies/float(torch.sum(target.data>0)+1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data[0] * mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss/s))

'''
...
epoch: 190 loss: tensor(0.9158)
epoch: 191 loss: tensor(0.9157)
epoch: 192 loss: tensor(0.9161)
epoch: 193 loss: tensor(0.9151)
epoch: 194 loss: tensor(0.9156)
epoch: 195 loss: tensor(0.9157)
epoch: 196 loss: tensor(0.9153)
epoch: 197 loss: tensor(0.9144)
epoch: 198 loss: tensor(0.9151)
epoch: 199 loss: tensor(0.9146)
epoch: 200 loss: tensor(0.9149)
'''

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(total_number_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = total_number_movies/float(torch.sum(target.data>0)+1e-10)
        test_loss +=np.sqrt(loss.data[0]*mean_corrector)
        s += 1.
print('test loss: ' + str(test_loss/s))

# test loss: tensor(0.9585)