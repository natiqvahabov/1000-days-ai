# -*- coding: utf-8 -*-

# importing Libraries: Pandas, Numpy, Pyplot, sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# first download csv from the web
import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
fetch_housing_data()

# load csv data
dataset = pd.read_csv("../data/housing.csv")

# there are 10 attributes
dataset.head()
dataset.info()

# ocean_proximity is categorical attribute
dataset['ocean_proximity'].value_counts()
'''
<1H OCEAN     9136
INLAND        6551
NEAR OCEAN    2658
NEAR BAY      2290
ISLAND           5
Name: ocean_proximity, dtype: int64
'''

# give initial info about each column, count mean std min max 25th 50th 75th percentiles
dataset.describe()

# draw histogram with pupulation column
dataset['population'].hist(bins=50, figsize=(20, 15))

# traing test split
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

# stratified sampling: the population is divided into homogeneous subgroups called strata
dataset['income_cat'] = np.ceil(dataset['median_income']/1.5)
dataset['income_cat'].where(dataset['income_cat']<5, 5.0, inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(dataset, dataset['income_cat']):
    strat_train_set = dataset.loc[train_index]
    strat_test_set = dataset.loc[test_index]

strat_test_set["income_cat"].value_counts() / len(strat_test_set)

# remove the income_cat attribute so the data is back to its original state
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# copying the data
ds_cp = strat_train_set.copy()

# visualize with high density by setting alpha variable
ds_cp.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)

# use colored map cmap jet
ds_cp.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=ds_cp["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()

# compute the standard correlation coefficient (also called Pearson’s r) 
#               between every pair of attributes using the corr() method

corr_matrix = ds_cp.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# another way look at correlation with pandas
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(ds_cp[attributes], figsize=(12, 8))
