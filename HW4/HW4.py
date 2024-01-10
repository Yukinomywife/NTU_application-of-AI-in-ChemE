# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 18:43:46 2023

@author: jimmy
"""

import os
os.chdir('C:/Users/jimmy/Desktop/AI app/HW4')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

np.random.seed(42)
df = pd.read_excel('1-s2.0-S240582972200352X-mmc1.xlsx')
df = df.drop(df.columns[-1],axis = 1)
df = df.drop(df.columns[0],axis = 1)
df = df.drop([0], axis = 0)
#df = df.dropna()
df = df.reset_index(drop = True)


y = df['Wet Thickness']
dff = pd.DataFrame(y)
X = df['Coating Gap (micron)']
dff.insert(len(dff.columns), column = "Coating Gap", value = X)
new_feature = df['Coating Gap (micron)'] ** 2
dff.insert(len(dff.columns), column = "Coating Gap^2", value = new_feature)
dff = dff.dropna()
dff = dff.reset_index(drop = True)

y = dff.iloc[:,0]
X = dff.iloc[:,1:len(dff.columns)]


model = LinearRegression()
model.fit(X, y)
score = model.score(X,y)
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(score, model.intercept_, model.coef_)
print(rmse)

model = RandomForestRegressor(n_estimators=5000, min_samples_leaf=2, min_samples_split=2)
model.fit(X, y)
y_pred_rf = model.predict(X)
score_rf = model.score(X,y)
rmse_rf = np.sqrt(mean_squared_error(y, y_pred_rf))
print(score_rf)
print(rmse_rf)


min_samples_leaf = 5
learning_rate = 0.005
n_estimators = 5000
min_weight_fraction_leaf = 0.01
model = GradientBoostingRegressor(min_samples_leaf =min_samples_leaf,learning_rate =learning_rate,n_estimators =n_estimators,min_weight_fraction_leaf =min_weight_fraction_leaf)
model.fit(X,y)
score_grad = model.score(X,y)
y_pred_grad = model.predict(X)
rmse_grad = np.sqrt(mean_squared_error(y, y_pred_grad))
print(score_grad)
print(rmse_grad)