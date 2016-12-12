###################### Import
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier)
from preprocess_01 import *
####################### RandomForest
RFC = RandomForestClassifier(n_jobs=2, n_estimators=10)
y_score = RFC.fit(x_train, np.array(y_train).astype(int))
predictions = RFC.predict(x_test)
print "*"*30
print "Accuracy de random forest"
print(accuracy_score(y_test, predictions))
print "*"*30
#Acurracy del 35%