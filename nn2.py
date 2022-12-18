#!/usr/bin/env python

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tensorflow import keras

df = pd.read_csv('matches.csv')
df = df.drop_duplicates()
df = df.dropna()
df = df.drop(['Unnamed: 0'], axis=1)

X = df.drop('Result', axis=1).values
y = df['Result'].values

test_data = pd.read_csv('./test_matches.csv')
test_data = test_data.drop_duplicates()
test_data = test_data.dropna()
test_data = test_data.drop(['Unnamed: 0'], axis=1)
test_results = test_data.Result
test_data = test_data.drop(['Result'], axis=1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

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
    print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')

predictions = model.predict(X_test)
predictions_test = model.predict(test_data)

predictions = (predictions > 0.5).astype(int)
predictions_test = (predictions_test > 0.5).astype(int)

print(f'Accuracy on test data: {accuracy_score(test_results, predictions_test)}')
