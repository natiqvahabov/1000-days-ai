#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 23:40:46 2018

@author: natig
"""
# Part1 Data Preprocessing

# importing ds libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing training set
train_dataset = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = train_dataset.iloc[:, 1:2].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))

training_set_scaled = sc.fit_transform(training_set)

# 60 timestemps and 1 output. each day stock price will look at 3 months previous data
# X_train 3 months data, Y_train next day stock

X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

# convert lists to np arrays
X_train, y_train = np.array(X_train), np.array(y_train)

# reshape dataset, keras needs (batch_size, timestemp, input_dim)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Part2 - Building the RNN

# importing keras libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# initialize regressor
regressor = Sequential()

# adding first LSTM layer, and Dropout regularization
# 3 inputs important: number_of_lstm_cells, return_sequence for several lstm layers, input_shape 3D
regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

# adding second LSTM layer, and Dropout regularization
regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

# adding third LSTM layer, and Dropout regularization
regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

# adding forth LSTM layer, and Dropout regularization
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# adding output layer
regressor.add(Dense(units=1))

# compiling the RNN, RMSprop is also good choice for RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# fitting RNN to the TrainingSet
regressor.fit(X_train, y_train, epochs=100, batch_size=32)


# Part3 - Making prediction and compare with original ones in visualized way
test_dataset = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_dataset.iloc[:, 1:2].values

# prediction
dataset_total = pd.concat((train_dataset['Open'], test_dataset['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(test_dataset) - 60:].values
inputs = sc.transform(inputs.reshape(-1,1))
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# visualize real and predicted 2017 January google stock price data
plt.plot(real_stock_price, color='red', label="Real stock price of Google")
plt.plot(predicted_stock_price, color='blue', label="Predicted stock price of Google")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.lagend()
plt.show()