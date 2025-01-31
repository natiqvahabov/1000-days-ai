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

# create new attributes
dataset["rooms_per_household"] = dataset["total_rooms"]/dataset["households"]
dataset["bedrooms_per_room"] = dataset["total_bedrooms"]/dataset["total_rooms"]
dataset["population_per_household"]=dataset["population"]/dataset["households"]

corr_matrix = dataset.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# take care of missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

# drop categorical data from dataset
housing_num = housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)
imputer.statistics_

X = imputer.transform(housing_num)

'''
SKLEARN Design well orginized:
    - Consistency (Estimators, Transformers, Predictors)
              imputer is estimator, imputer.fit, imputer.transform, imputer.predict
    - Inspection (imputer.strategy, imputer.statistics_)
    - Nonproliferation of classes
    - Composition
    - Sensible defaults.
'''


# Handling Text and Categorical Attributes
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)

# conver text to numbers by using OrdinalEncoder
from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()
housing_cat_encoded = ord_enc.fit_transform(housing_cat)
housing_cat_encoded[:10]

# onehotencoder
# toarray() convert sparse matrix to numpy array
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder()
housing_cat_1hot = one_hot_encoder.fit_transform(housing_cat).toarray()

# feature scaling
#There are two common ways to get all attributes to have the same scale:
#                       min-max scaling and standardization.
# min-max scaling is also called normalization
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
housing_num_tr = num_pipeline.fit_transform(housing_num)

# ColumnTransformer
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)

# Training the model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = dataset.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))

# calculate mean squared error
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
# 69050
# RESULT SHOW UNDERFITTING BECAUSE OF NOT ACCURACY RESUL, WE SHOULD USE COMPLICATED MODEL

# Let's use Decision Tree, non-linear model
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

# make prediction with dec_tree
housing_predictions = tree_reg.predict(housing_prepared)

# rmse with dec_tree
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
# 0.0
# IT SHOWED OVERFITTING RESULT.

# use 10-fold cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

'''
Scikit-Learn’s cross-validation features expect a utility function (greater is better)
rather than a cost function (lower is better), so the scoring function is actually
the opposite of the MSE (i.e., a negative value), which is why the preceding 
code computes -scores before calculating the square root.
'''

tree_rmse_scores.mean()
# 70033
tree_rmse_scores.std()
# 2655

# Decision Tree gave much more bad result than Linear Regression
# now let's use random forest

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)

# 21806 
# That gaved much more accurate result rather 2 previous models: Linear and DTree

# Using GridSearchCV for finding best hyperparameters, there is also RandomSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_
# {'max_features': 8, 'n_estimators': 30}

grid_search.best_estimator_
'''
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features=8, max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=30, n_jobs=None, oob_score=False,
           random_state=None, verbose=0, warm_start=False)
'''

# look at each result
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
'''
64446.92237885503 {'max_features': 2, 'n_estimators': 3}
55111.01664184465 {'max_features': 2, 'n_estimators': 10}
52864.16217859041 {'max_features': 2, 'n_estimators': 30}
60550.46532054315 {'max_features': 4, 'n_estimators': 3}
52780.00459507404 {'max_features': 4, 'n_estimators': 10}
50871.9101975998 {'max_features': 4, 'n_estimators': 30}
58952.24415819549 {'max_features': 6, 'n_estimators': 3}
52943.89258986576 {'max_features': 6, 'n_estimators': 10}
50210.987226964084 {'max_features': 6, 'n_estimators': 30}
58226.967896915376 {'max_features': 8, 'n_estimators': 3}
52657.386082237244 {'max_features': 8, 'n_estimators': 10}
50207.38057065243 {'max_features': 8, 'n_estimators': 30}
62367.54070316257 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
54088.88616936969 {'bootstrap': False, 'max_features': 2, 'n_estimators': 10}
60529.84778260889 {'bootstrap': False, 'max_features': 3, 'n_estimators': 3}
52841.96575684903 {'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
59692.12892015442 {'bootstrap': False, 'max_features': 4, 'n_estimators': 3}
52246.603283102144 {'bootstrap': False, 'max_features': 4, 'n_estimators': 10}
'''

# Analyize the features and their importance
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

# evaluate the test result
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
# 47135

# checking 95% confidence interval
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                             loc=squared_errors.mean(),
                             scale=stats.sem(squared_errors)))
# array([45197.75776524, 48996.4404598 ])