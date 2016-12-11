###################### Import

import csv
import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix

from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.preprocessing import (StandardScaler, OneHotEncoder, Imputer)
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, roc_curve, auc)
from sklearn.linear_model import (LogisticRegression, SGDClassifier, Perceptron, PassiveAggressiveClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.cluster import KMeans
from sklearn import preprocessing

####################### Preprocessing

all_train = pd.read_csv('./data/train.csv')
#Quitamos las nulas
all_train = all_train[(all_train.Upc.notnull()) & (all_train.DepartmentDescription.notnull()) ]
#Dividimos entre entrenamiento y prueba
all_train['is_train'] = np.random.uniform(0, 1, len(all_train)) <= .75
le = preprocessing.LabelEncoder()
#De categoricas a numero Weekday y Departamento
all_train['Weekday'] = le.fit_transform(all_train['Weekday'])
le = preprocessing.LabelEncoder()
all_train['DepartmentDescription'] = le.fit_transform(all_train['DepartmentDescription'])
del all_train['FinelineNumber']
del all_train['Upc']
del all_train['VisitNumber']


#Dividimos
#Datos sin escalar
train, test = all_train[all_train['is_train']==True], all_train[all_train['is_train']==False]

del train ['is_train']
del test ['is_train']
# Escalamos variales
scaler = preprocessing.StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(all_train), columns=all_train.columns)
df_scaled['is_train'] = all_train['is_train']

train_scaled, test_scaled = df_scaled[df_scaled['is_train']==True], df_scaled[df_scaled['is_train']==False]

del train_scaled ['is_train']
del test_scaled ['is_train']

x_train_scaled = train_scaled[train_scaled.columns[1:]].values
y_train_scaled = train_scaled[train_scaled.columns[0]].values
x_test_scaled = test_scaled[test_scaled.columns[1:]].values
y_test_scaled = test_scaled[test_scaled.columns[0]].values



x_train = train[train.columns[1:]].values
y_train = train[train.columns[0]].values
x_test = test[test.columns[1:]].values
y_test = test[test.columns[0]].values


