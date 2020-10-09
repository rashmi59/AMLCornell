#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 19:03:29 2020

@author: rashmisinha
"""
import pandas as pd
import numpy as mp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import LabelEncoder

#reading training data
train_data = pd.read_csv("train.csv")
#getting columns with incomplete training data
nacoltrain = train_data.columns[train_data.isna().any()].tolist()

#reading test data
test_data = pd.read_csv("test.csv")
#getting columns with incomplete test data
nacoltest = test_data.columns[test_data.isna().any()].tolist()

#excluding columns with incomplete data plus a few additional data
X_train  = train_data.drop(nacoltrain + nacoltest + ['Id', 'SalePrice'], axis=1)
X_test  = test_data.drop(nacoltrain + nacoltest + ['Id'], axis=1)

#getting all headers
headers = X_test.columns.values.tolist()

#getting headers with numerical labels
numerical_headers = ['LotArea','1stFlrSF','2ndFlrSF','LowQualFinSF',
               'GrLivArea','FullBath','HalfBath','BedroomAbvGr',
               'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','WoodDeckSF',
               'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch',
               'PoolArea']

#performing encoding for categorical variables
for col in [i for i in headers if i not in numerical_headers]:
    X_train[col] = LabelEncoder().fit_transform(X_train[col])
    X_test[col] = LabelEncoder().fit_transform(X_test[col])

#getting output data for training set
Y_train = train_data['SalePrice']

#performing linear regression
linear_regr = LinearRegression().fit(X_train, Y_train)
Y_pred_train = linear_regr.predict(X_train)
#computing error rate for training data
k1 = ((abs(Y_pred_train - Y_train)/Y_train).sum())/Y_train.shape[0] * 100

#predicting for test data
Y_test_linear = linear_regr.predict(X_test)
Y_test_linear_df = pd.DataFrame(Y_test_linear, columns=['SalePrice'])
Y_test_linear_df.insert(0,'Id',test_data['Id'])
Y_test_linear_df.to_csv(r'LinearRegression.csv')


#Ridge regression(L2 method)
ridge_regr = Ridge(alpha = 1).fit(X_train, Y_train)
Y_pred_ridge_train = ridge_regr.predict(X_train)
#computing error rate for training data
k2 = ((abs(Y_pred_ridge_train - Y_train)/Y_train).sum())/Y_train.shape[0] * 100

#predicting for test data
Y_test_ridge = ridge_regr.predict(X_test)
Y_test_ridge_df = pd.DataFrame(Y_test_ridge, columns=['SalePrice'])
Y_test_ridge_df.insert(0, 'Id', test_data['Id'])
Y_test_ridge_df.to_csv(r'RidgeRegression.csv')

#Lasso regression(L1 method)ß
lasso_regr = Lasso(alpha = 0.05).fit(X_train, Y_train)

#predicting for test data
Y_test_lasso = lasso_regr.predict(X_test)
Y_test_lasso_df = pd.DataFrame(Y_test_lasso, columns=['SalePrice'])
Y_test_lasso_df.insert(0, 'Id', test_data['Id'])
Y_test_lasso_df.to_csv(r'LassoRegression.csv')