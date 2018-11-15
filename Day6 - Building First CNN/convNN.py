#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:59:25 2018

@author: natig
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

# initialize CNN
classifier = Sequential()

# Step1 Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# Step2 Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step3 Flatten
classifier.add(Flatten())

# Step4 Full-connection
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

# compile classifier
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])