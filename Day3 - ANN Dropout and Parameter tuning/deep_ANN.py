#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  11 14:24:14 2018

@author: natig
"""

# ------------------------   Day1   -----------------------------

# importing python library for csv table reading - pandas
import pandas as pd

# loading csv table
dataset = pd.read_csv('data/bank_customer_details.csv')
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
X_test = sc.transform(X_test) 

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

cm = confusion_matrix(y_test,y_pred.round())
# true outcomes of cm were 84.2 % of all test data 
# 1684 out of 2000


# ------------------------   Day2   -----------------------------

# predict new entered row
new_node_X = [0,0,619,1,40,3,60000,2,1,1,50000]
new_node_X = sc.transform(np.reshape(new_node_X,(11,1)).T)
new_row_pred = classifier.predict(new_node_X)
new_row_pred = (new_row_pred>0.5)
print(new_row_pred) # False

# Evaluating the ANN, 10 fold cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, init='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, init='uniform', activation='relu'))
    classifier.add(Dense(units=1, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10,n_jobs=1)
mean = accuracies.mean() # 0.8352499946579337
variance = accuracies.std() # 0.012634278179076582


# ------------------------   Day3   -----------------------------

from keras.layers import Dropout

#adding dropout to first 2 hidden layers
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, init='uniform', activation='relu', input_dim=11))
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(units=6, init='uniform', activation='relu'))
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(units=1, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
