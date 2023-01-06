#!/usr/bin/env python3
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import pandas as pd

from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split

data = pd.read_csv('./datasets/matches.csv')

test_data = pd.read_csv('./datasets/test_matches.csv')
test_data = test_data.drop_duplicates()
test_data = test_data.dropna()
test_data = test_data.drop(['Unnamed: 0'], axis=1)
test_results = test_data.Result
test_data = test_data.drop(['Result'], axis=1)

data = data.reset_index(drop=True)
data = data.drop_duplicates()
data = data.dropna()
data = data.drop(['Unnamed: 0'], axis=1)
results = data.Result
data = data.drop(['Result'], axis=1)
print(f'Lenght of data: {len(data)}')

print('----------------------')
print('----------------------')

x_train, x_test, y_train, y_test = train_test_split(data, results)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

predicted_rfc = rfc.predict(x_test)
predicted_rfc_test = rfc.predict(test_data)
from sklearn.metrics import accuracy_score
print(f'Full Test RFC: {accuracy_score(predicted_rfc_test, test_results)}')
print(f'Split Test RFC: {accuracy_score(predicted_rfc, y_test)}')

print('----------------------')
print('----------------------')

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

predicted_clf = clf.predict(x_test)
predicted_clf_test = clf.predict(test_data)
print(f'Full Test DTC: {accuracy_score(predicted_clf_test, test_results)}')
print(f'Split Test DTC: {accuracy_score(predicted_clf, y_test)}')
