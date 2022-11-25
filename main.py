#imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sklearn.metrics as metrics
import os

#train and test data
PATH = 'data/'
train = os.path.join(PATH, 'train.csv')
test = os.path.join(PATH, 'test.csv')

#read data from csv file
df_train = pd.read_csv(train)
df_test = pd.read_csv(test)

print(df_train['SalePrice'].describe())

#histogram
plt.hist(df_train['SalePrice'], bins=50)
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.show()

#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
plt.show()

#log transform the target variable
df_train['SalePrice'] = np.log(df_train['SalePrice'])
plt.show()

#delete outliers
df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice'] < 300000)].index)
fig, ax = plt.subplots()
ax.scatter(df_train['GrLivArea'], df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

#missing data
y_train = df_train['SalePrice']
test_id = df_test['Id']
all_data = pd.concat([df_train, df_test], axis=0, sort=False)
all_data = all_data.drop(['Id', 'SalePrice'], axis=1)

Total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum() / all_data.isnull().count()*100).sort_values(ascending=False)
missing_data = pd.concat([Total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(25))

#fill numeric values
numeric_missed = [
      'BsmtFinSF1', 
      'BsmtFinSF2', 
      'BsmtFullBath', 
      'BsmtHalfBath', 
      'BsmtUnfSF', 
      'GarageArea', 
      'GarageCars', 
      'GarageYrBlt',
]

for feature in numeric_missed:
      all_data[feature] = all_data[feature].fillna(0)

#fill categorical values
categorical_missed = [
      'Exterior1st',
      'Exterior2nd',
      'Functional',
      'KitchenQual',
      'MSZoning',
      'SaleType',
]

for feature in categorical_missed:
      all_data[feature] = all_data[feature].fillna(all_data[feature].mode()[0])

all_data['Functional'] = all_data['Functional'].fillna('Typ')
all_data.drop(['Utilities'], axis = 1, inplace=True)
all_data.isnull().sum().max()


#convert categorical variable into dummy/indicator variables
all_data = pd.get_dummies(all_data)
print(all_data.head())

#split data
x_train = all_data[:len(y_train)]
x_test = all_data[len(y_train):]

print(f'x_test and x_train shape: {x_test.shape, x_test.shape}')

#model
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

param_grid = [
      {
            'subsample': [0.45, 0.5, 0.55], 'n_estimators': [1200, 1400],
            'max_depth': [5], 'learning_rate': [0.02],
            'colsample_bytree': [0.4], 'colsample_bylevel': [0.5],
            'reg_alpha': [1], 'reg_lambda': [1], 'min_child_weight': [2],
      }
]

xgb_model = XGBRegressor(eval_metric='rmse')
grid_search = GridSearchCV(
      xgb_model, 
      param_grid = param_grid,
      scoring='neg_mean_squared_error', 
      n_jobs=10,
      cv=5,
      verbose=True
)

grid_search.fit(x_train, y_train)
print(grid_search.best_score_)

#predict
y_predict = np.floor(np.expm1(grid_search.best_estimator_.predict(x_test)))
print(y_predict)

sub = pd.DataFrame()
sub['Id'] = test_id
sub['SalePrice'] = y_predict
sub.to_csv('submission.csv', index=False)
