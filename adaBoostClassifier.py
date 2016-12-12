from sklearn.ensemble import *
from sklearn.metrics import accuracy_score


# Ensemble algorithms used

model_adaBoost = AdaBoostClassifier(n_estimators=100)
model_adaBoost.fit(x_train, y_train)
predictions_ada = model_adaBoost.predict(x_test)
print(accuracy_score(y_test, predictions_ada))
# 30% de accuracy

