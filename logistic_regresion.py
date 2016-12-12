
###################### Import

from preprocess_01 import *
####################### RandomForest
num_folds = 10
n_estimators = 100
num_instances = len(x_train)
seed = 7
scoring = 'accuracy'

LR = LogisticRegression(solver='sag', tol=1e-1, C=1.e5 / x_train.shape[0])
model = LR
model.fit(x_train_scaled, y_train)
predictions =  model.predict(x_test_scaled)


print "*"*30
print  "%s" % (accuracy_score(y_test, predictions))
print "*"*30
