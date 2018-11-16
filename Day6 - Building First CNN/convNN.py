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

# adding second convolutional layer to reduce difference accuracy between train and test model
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step3 Flatten
classifier.add(Flatten())

# Step4 Full-connection
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

# compile classifier
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fitting the cnn with images, image augmentation to get rid of overfitting
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

# For large epoches we should have GPU, with limited CPU I dared for only 2 epoches, it took near 2 hours.
# Epoch 1/2
# 8000/8000 [==============================] - 2522s 315ms/step - loss: 0.3534 - acc: 0.8332 - val_loss: 0.5892 - val_acc: 0.8030
# Epoch 2/2
# 7999/8000 [============================>.] - ETA: 0s - loss: 0.1012 - acc: 0.9620


# predict individual test
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

# DOG - correct

