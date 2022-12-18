#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('./matches.csv')
data = data.drop_duplicates()
data = data.dropna()
data = data.drop(['Unnamed: 0'], axis=1)
results = data.Result
data = data.drop(['Result'], axis=1)
print(f'Lenght of data: {len(data)}')

x_train, x_test, y_train, y_test = train_test_split(data, results)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

predicted_rfc = rfc.predict(x_test)
from sklearn.metrics import accuracy_score
print(f'RFC: {accuracy_score(predicted_rfc, y_test)}')

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

predicted_clf = clf.predict(x_test)
print(f'CLF: {accuracy_score(predicted_clf, y_test)}')

# In[314]:

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)

predicted_svc = svc.predict(x_test)
print(f'SVC: {accuracy_score(predicted_svc, y_test)}')
