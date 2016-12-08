from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt
import pandas as pd
import numpy as np
from sklearn import preprocessing

dataset = pd.read_csv('./train.csv')
dataset = dataset[(dataset.Upc.notnull()) & (dataset.DepartmentDescription.notnull()) ]
dataset['is_train'] = np.random.uniform(0, 1, len(dataset)) <= .75
le = preprocessing.LabelEncoder()
dataset['Weekday'] = le.fit_transform(dataset['Weekday'])
le = preprocessing.LabelEncoder()
dataset['DepartmentDescription'] = le.fit_transform(dataset['DepartmentDescription'])
train, test = dataset[dataset['is_train']==True], dataset[dataset['is_train']==False]

features = dataset.columns[1:-1]
y = dataset.columns[0]

rf = RandomForestClassifier(n_estimators=100)

rf.fit(train[features], train[y])
