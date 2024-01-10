# -*- coding: utf-8 -*-
"""
Created on Wed May  3 12:40:46 2023

@author: jimmy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,BaggingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
import random
from sklearn.linear_model import Ridge

random.seed(42)
np.random.seed(42)

df = pd.read_excel('MOFs_Kh_and_structural_properties.xlsx')

y = df.iloc[:, 1].values


ratio_dia = df['PLD (A)']/df['LCD (A)']
ratio_dia = np.exp(0.8098801 * ratio_dia)
df['LCD (A)'] = np.log(df['LCD (A)'])
df['PLD (A)'] = np.log(df['PLD (A)'])
df['Density (kg/m3)'] = np.log(df['Density (kg/m3)'])
df['Surface Area (m2/g)'] = np.exp(0.00016638 * df['Surface Area (m2/g)'])
df['Void Fraction (-)'] = np.log(df['Void Fraction (-)'])
df['Void Volume (cm3/g)'] = np.log(df['Void Volume (cm3/g)'])

df.insert(5, column = "PLD/LCD", value = ratio_dia)

df = pd.get_dummies(df[df.columns[3:]])

X = df.iloc[:,:].values

scaler = StandardScaler()
y = scaler.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2858)

#lin = Ridge(random_state = 42)
#lin.fit(X_train, y_train.ravel())
forest = RandomForestRegressor(n_estimators = 300,random_state = 42, n_jobs=-1)
forest.fit(X_train, y_train.ravel())
#bag = BaggingRegressor(estimator=forest,n_jobs=-1,random_state=42)
#bag.fit(X_train, y_train.ravel())
#ada = AdaBoostRegressor(estimator=forest, random_state=42)
#ada.fit(X_train, y_train.ravel())

cv = KFold(n_splits=5, shuffle=True, random_state=1)
scores = cross_val_score(forest, X, y.ravel(), scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
print('Mean RMSE: %.3f (%.3f)' % (np.mean(np.sqrt(-scores)), np.std(np.sqrt(-scores))))


y_predict = forest.predict(X_test)
acc_0 = forest.score(X_train, y_train)
print(acc_0)
acc = forest.score(X_test, y_test)
print(acc)
y_plot = y_test

from sklearn.metrics import r2_score
def adjusted_r2_score(y_test, y_predict, n_features):
    r2 = r2_score(y_test, y_predict)
    adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - n_features - 1)
    return adj_r2

n_features = X.shape[1]
adj_r2 = adjusted_r2_score(y_test, y_predict, n_features)
print("Adjusted R-squared:", adj_r2)

plt.plot(y_test, y_predict, "b^",y_test,y_plot)
plt.show