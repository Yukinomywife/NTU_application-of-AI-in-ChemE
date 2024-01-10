# -*- coding: utf-8 -*-
"""
Created on Wed May  3 12:40:46 2023

@author: jimmy
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import r2_score
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l1, l2, l1_l2
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
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

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,restore_best_weights=True)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(18384,input_shape=(X_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4096),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2048),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu',\
                            kernel_regularizer=l2(0.25),\
                            bias_regularizer =l1(0.25),\
                            activity_regularizer = l1_l2(l1=0.01, l2=0.01)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1)
])

optimizer = Adam(learning_rate=1e-5)

model.compile(optimizer=optimizer, loss='mse')

model.fit(X_train, y_train,validation_split=0.05,\
          epochs=1600,verbose=0,batch_size=256,callbacks=[callback])

y_pred = model.predict(X_test,batch_size=1,verbose=1)

r2 = r2_score(y_test.reshape(-1,1), y_pred)
print(r2)

model.save('zeolite_model.h5')