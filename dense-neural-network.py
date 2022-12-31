#!/usr/bin/env python3.8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Do not show Warning() messages

import tensorflow as tf
tf.get_logger().setLevel('ERROR') # Do not show Warning() messages if any, only show errors

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras

df = pd.read_csv('./datasets/matches.csv')
df = df.drop(['Unnamed: 0'], axis=1)
df = df.drop_duplicates()
df = df.dropna()

X = df.drop('Result', axis=1).values
y = df['Result'].values

test_data = pd.read_csv('./datasets/test_matches.csv')
test_data = test_data.drop_duplicates()
test_data = test_data.dropna()
test_data = test_data.drop(['Unnamed: 0'], axis=1)
test_results = test_data.Result
test_data = test_data.drop(['Result'], axis=1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1), loss='mean_squared_error', metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

predictions = model.predict(X_test)
predictions_test = model.predict(test_data)
