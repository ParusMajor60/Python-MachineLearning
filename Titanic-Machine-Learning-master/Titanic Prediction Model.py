# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 23:33:38 2018

@author: dell
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split

# Just to switch off pandas warning
pd.options.mode.chained_assignment = None

data = pd.read_csv("titanic_train.csv")
print(data.head())

data.columns

median_age = data['age'].median()
print('Median age is {}'.format(median_age))

data['age'].fillna(median_age,inplace=True)

data_inputs = data[['pclass','age','sex']]
print(data_inputs.head())

expected_output = data[['survived']]
print(expected_output.head())

data_inputs['pclass'].replace('3rd',3,inplace=True)
data_inputs['pclass'].replace('2nd',2,inplace=True)
data_inputs['pclass'].replace('1st',1,inplace=True)
print(data_inputs.head())

data_inputs['sex'] = np.where(data_inputs['sex']=='female',0,1)
#data_inputs['sex'].replace('female',0,inplace=True)
#data_inputs['sex'].replace('male',1,inplace=True)
print(data_inputs.head())


# =============================================================================
# inputs_train, inputs_test, expected_output_train, expected_output_test = train_test_split(data_inputs,expected_output,test_size = 0.33, random_state = 42)
# print(inputs_train.head())
# print(expected_output_train.head())
# =============================================================================

rf = RandomForestClassifier(n_estimators=1000)

rf.fit(data_inputs,expected_output)

data_test = pd.read_csv('titanic_test.csv')
data_test['age'].fillna(median_age,inplace=True)
inputs_test = data_test[['pclass','age','sex']]
expected_output_test = data_test[['survived']]
inputs_test['pclass'].replace('3rd',3,inplace=True)
inputs_test['pclass'].replace('2nd',2,inplace=True)
inputs_test['pclass'].replace('1st',1,inplace=True)
inputs_test['sex'] = np.where(inputs_test['sex']=='female',0,1)

accuracy1 = rf.score(inputs_test, expected_output_test)
print('Accuracy = {}%'.format(accuracy1*100))

lr = LogisticRegression()
lr.fit(data_inputs,expected_output)
accuracy2 = lr.score(inputs_test, expected_output_test)
print('Accuracy = {}%'.format(accuracy2*100))