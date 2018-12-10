# MNIST dataset is Hello World of Classification Problems
#       which is a set of 70,000 small images of digits handwritten
#               by high school students and employees of the US Census Bureau

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
mnist

X, y = mnist["data"], mnist["target"]
X.shape
y.shape

# look at individual image
import matplotlib
import matplotlib.pyplot as plt

rnd_dig = X[2315]
rnd_dig_image = rnd_dig.reshape(28, 28)
plt.imshow(rnd_dig_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()
# 0 which is equal to y[2315]

# train, test split
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# shuffle train set
import numpy as np
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# train model for one digit - 2, it is binary classifier, set True to column
# which stores 2, and False to other numbers
y_train_2 = (y_train==2)

from sklearn.linear_model import SGDClassifier
sgd_clsf = SGDClassifier(random_state=42)
sgd_clsf.fit(X_train, y_train_2)

# predict test result
sgd_clsf.predict([X_test[2142]])
# array([True])

# Performance measure
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clsf, X_train, y_train_2, cv=3, scoring="accuracy")
# array([0.96995, 0.9712 , 0.974  ])
# 96% accuracy

# train and fit model for not 2 members
from sklearn.base import BaseEstimator
class Never2Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_2_clsf = Never2Classifier()
cross_val_score(never_2_clsf, X_train, y_train_2, cv=3, scoring="accuracy")
# array([0.89955, 0.90205, 0.9005 ]) - 89% accuracy
# it means if we just throw False to each sample, it would give us 89% accuracy
# so accuracy is not good performance measure for classification problems

# Confusion Matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clsf, X_train, y_train_2, cv=3)
confusion_matrix(y_train_2, y_train_pred)
# 34.35651149086623

# precision & recall
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_2, y_train_pred)
# 0.8935895067430261, 89% accuracy for predicting 2s
recall_score(y_train_2, y_train_pred)
# 0.8118496139644176, 81% accuracy for predicting non-2s

# calculationg F1 score, harmonic mean of recall and precision
from sklearn.metrics import f1_score
f1_score(y_train_2, y_train_pred)
# 0.8507607070618239

'''
Unfortunately, you can’t have it both ways: increasing precision reduces recall,
             and vice versa. This is called the precision/recall tradeoff.
'''