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
movies  = pd.read_csv('../data/boltzman_machine/ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users   = pd.read_csv('../data/boltzman_machine/ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('../data/boltzman_machine/ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

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