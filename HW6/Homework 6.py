# -*- coding: utf-8 -*-
"""
Created on Thu May 18 12:34:29 2023

@author: jimmy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

f = pd.read_csv('psca_prop.csv')
# f = pd.read_csv('p_prop.csv')

# only use when psca
for i in range(len(f)):
    if f.scaffold[i] != 'O=C1C=CCO1':
        f = f.drop([i],axis = 0)
        
logP = f.iloc[:,2]
tPSA = f.iloc[:,3]
QED = f.iloc[:,4]

rel_freq1 = logP.value_counts(normalize=True,ascending=True)
rel_freq2 = tPSA.value_counts(normalize=True,ascending=True)
rel_freq3 = QED.value_counts(normalize=True,ascending=True)

rel_freq1 = rel_freq1.sort_values()

plt.bar(rel_freq1.index, rel_freq1.values * 100,width = 0.3)#, linestyle='-')
plt.xlabel('logP')
plt.xticks(np.arange(0, 5.5,0.5))
plt.ylabel('Percentage')
plt.title('Probability Distribution of logP')
plt.show()

rel_freq2 = rel_freq2.sort_values()

plt.bar(rel_freq2.index, rel_freq2.values * 100,width = 1.2)#, linestyle='-')
plt.xlabel('tPSA')
plt.xticks(np.arange(0, 210,20))
plt.ylabel('Percentage')
plt.title('Probability Distribution of tPSA')
plt.show()

rel_freq3 = rel_freq3.sort_values()

plt.bar(rel_freq3.index, rel_freq3.values * 100,width = 0.2)#, linestyle='-')
plt.xlabel('QED')
plt.xticks(np.arange(0, 1.7,0.1))
plt.ylabel('Percentage')
plt.title('Probability Distribution of QED')
plt.show()

import numpy as np
from sklearn.metrics import mean_squared_error

predicted = logP
actual = np.full(len(logP),2.5)
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)
print("RMSE of logP:", rmse)

predicted = tPSA
actual = np.full(len(tPSA),100)
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)
print("RMSE of tPSA:", rmse)

predicted = QED
actual = np.full(len(QED),0.8)
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)
print("RMSE of QED:", rmse)