# -*- coding: utf-8 -*-
"""
Created on Wed May  3 12:40:46 2023

@author: jimmy
"""

import os
os.chdir('C:/Users/jimmy/Desktop/AI app/HW5')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

# read the datafile
df = pd.read_excel('MOFs_Kh_and_structural_properties.xlsx')
df = df.drop(["Unnamed: 2"],axis =1)
values = [0]
df = df[df.isin(values) == False]
df = df.dropna()
df = df.reset_index(drop = True)

y = df.iloc[:, 1].values
df['LCD (A)'] = np.log(df['LCD (A)'])
df['PLD (A)'] = np.log(df['PLD (A)'])
df['Density (kg/m3)'] = np.log(df['Density (kg/m3)'])
df['Surface Area (m2/g)'] = np.log(df['Surface Area (m2/g)'])
df['Void Fraction (-)'] = np.log(df['Void Fraction (-)'])
df['Void Volume (cm3/g)'] = np.log(df['Void Volume (cm3/g)'])
ratio_dia = df['PLD (A)']/df['LCD (A)']
df.insert(5, column = "PLD/LCD", value = ratio_dia)
df = pd.get_dummies(df[df.columns[2:]])

X_1 = df.iloc[:, :7].values
X_2 = df.iloc[:,7:].values
y = np.log(y)

scaler = StandardScaler()
X_value = scaler.fit_transform(X_1[:,:])
X = np.concatenate([X_value,X_2],axis = 1)
y = scaler.fit_transform(y.reshape(-1, 1))
'''
# randomly (based on the random_state) split the dateset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)

forest = RandomForestRegressor(random_state = 42)
forest.fit(X_train, y_train.ravel())
# 1ada = AdaBoostRegressor(base_estimator=forest, random_state=42)
# ada.fit(X_train, y_train)
y_predict = forest.predict(X_test)
acc = forest.score(X_test, y_test)
print(acc)
# y_plot = y_test
# plt.plot(y_test, y_predict, "b^",y_test,y_plot)
# plt.show
'''

# import time
# start_time = time.time()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# # from sklearn.inspection import permutation_importance

# np.random.seed(42)
# # read the datafile
# df = pd.read_excel('MOFs_Kh_and_structural_properties.xlsx')
# X_value = df.iloc[0:8814, 3:9]
# y = df.iloc[0:8814, 1].values

# OMS = pd.get_dummies(df['Has_OMS'])  #-> Yes,No
# MeSite = pd.get_dummies(df['Open_Metal_Sites']) #very large
# AllMe = pd.get_dummies(df['All_Metals']) #supa largee
# ratio_dia = df['LCD (A)']/df['PLD (A)']
# X_value.insert(len(X_value.columns)-1, column = "LCD/PLD", value = ratio_dia)

# X = pd.concat([X_value,OMS,MeSite,AllMe],axis=1)
# X = X.iloc[0:8814,:].values

# # randomly (based on the random_state) split the dateset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# forest = RandomForestRegressor(random_state = 42, min_samples_split = 2)
# forest.fit(X_train, y_train)
# y_predict = forest.predict(X_test)
# acc = forest.score(X_test, y_test)
# print(acc)
# y_plot = y_test
# plt.plot(y_test, y_predict, "b^",y_test,y_plot)
# plt.show

# #    
# end_time = time.time()
# elapsed_time = end_time - start_time

# print("Elapsed time:", elapsed_time, "seconds.")
