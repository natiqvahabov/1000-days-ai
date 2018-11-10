#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 12:54:24 2018

@author: natig
"""
# importing python library for csv table reading - pandas
import pandas as pd

# loading csv table
dataset = pd.read_csv('bank_customer_details.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# encoding categorical values to numerical
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()

# encode country variable
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# encode gender variable
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

# split country to each column 0 and 1 to avoid dummy variable trap
onehotencoder_1 = OneHotEncoder(categorical_features = [1])
X = onehotencoder_1.fit_transform(X).toarray()

# remove first column of table due to DVT
X = X[:,1:]

# test train split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling, we need to do it , to not allow to a feature dominate others
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test) 

# keras
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Adding output layer
classifier.add(Dense(units=1, kernel_initializer='uniform',activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
# classifier received 83.4 % average correctness after 100 epoches

y_pred = classifier.predict(X_test)

# confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
# true outcomes of cm were 84.2 % of all test data 
# 1684 out of 2000
