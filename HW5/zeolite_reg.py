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
from sklearn.linear_model import LinearRegression,Ridge

df = pd.read_excel('Zeolites_Kh_and_structural_properties.xlsx')

df = df.drop(["Unnamed: 2"],axis =1)
values = [0]
df = df[df.isin(values) == False]
df = df.dropna()
df = df.reset_index(drop = True)

df['Di (A)'] = np.log(df['Di (A)'])
df['Df (A)'] = np.log(df['Df (A)'])
df['Surface Area (m2/g)'] = np.log(df['Surface Area (m2/g)'])
df['Void Fraction (-)'] = np.log(df['Void Fraction (-)'])
df['Void Volume (cm3/g)'] = np.log(df['Void Volume (cm3/g)'])

X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values
y = np.log(y)

scaler = StandardScaler()
y = scaler.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1973)

#lin = LinearRegression(n_jobs=-1)
#lin = Ridge(random_state = 42)
#lin.fit(X_train, y_train.ravel())
forest = RandomForestRegressor(n_estimators = 1000,random_state = 42, n_jobs=-1)
#bag = BaggingRegressor(estimator=forest,n_jobs=-1,random_state=42)
#ada = AdaBoostRegressor(estimator=bag, random_state=42)
forest.fit(X_train, y_train.ravel())
#bag.fit(X_train, y_train.ravel())
#ada.fit(X_train, y_train.ravel())

cv = KFold(n_splits=5, shuffle=True, random_state=1)
scores = cross_val_score(forest, X, y.ravel(), scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
print('RMSE: %.3f (%.3f)' % (np.mean(np.sqrt(-scores)), np.std(np.sqrt(-scores))))

y_predict = forest.predict(X_test)
acc_0 = forest.score(X_train, y_train.ravel())
print(acc_0)
acc = forest.score(X_test, y_test)
print(acc)
y_plot = y_test
plt.plot(y_test, y_predict, "b^",y_test,y_plot)
plt.show

from sklearn.metrics import r2_score
def adjusted_r2_score(y_test, y_predict, n_features):
    r2 = r2_score(y_test, y_predict)
    adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - n_features - 1)
    return adj_r2

n_features = X.shape[1]
adj_r2 = adjusted_r2_score(y_test, y_predict, n_features)
print("Adjusted R-squared:", adj_r2)