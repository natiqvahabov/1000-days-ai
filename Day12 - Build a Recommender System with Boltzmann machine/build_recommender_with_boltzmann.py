#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 11:15:09 2018

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

# convert non-rating movies to 0, unliked(1,2) to 0, liked(3,4,5) to 1
training_set[training_set==0] = -1
training_set[training_set==1] = 0
training_set[training_set==2] = 0
training_set[training_set>=3] = 1

test_set[test_set==0] = -1
test_set[test_set==1] = 0
test_set[test_set==2] = 0
test_set[test_set>=3] = 1

# Creating architecture of the Neural Network
# RBM is probibility based model

# nv - number of visible nodes, nh - number of hidden nodes
# W - weights, a - bias for hidden, a- bias for visible nodes
#  
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    def sample_h(self, x):     
        # x is visible nodes v in p_h_given_v
        # self.W.t() transpose function of weights     
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):     
        # y is visible nodes v in p_v_given_h     
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h )
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
nv = len(training_set[0])
nh = 100
batch_size=100
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch+1):
    train_loss = 0
    s = 0.
    for id_user in range(0, total_number_users - batch_size, batch_size):
        vk = training_set[id_user:id_user + batch_size]
        v0 = training_set[id_user:id_user + batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            # Markov Chain Monta Carlo method
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>0] - vk[v0>0]))
        s += 1.
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss/s))

'''       
#epoch: 1 loss: tensor(0.3171)
#epoch: 2 loss: tensor(0.1559)
#epoch: 3 loss: tensor(0.1490)
#epoch: 4 loss: tensor(0.1522)
#epoch: 5 loss: tensor(0.1511)
#epoch: 6 loss: tensor(0.1531)
#epoch: 7 loss: tensor(0.1459)
#epoch: 8 loss: tensor(0.1480)
#epoch: 9 loss: tensor(0.1489)
#epoch: 10 loss: tensor(0.1476)
'''

# Testing the RBM
test_loss = 0
s = 0
for id_user in range(total_number_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0])>0:
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>0]-v[vt>0]))
        s+=1
print('test loss: '+str(test_loss/s))

# test loss: tensor(0.1672)