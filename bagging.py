__author__ = 'tim'
from sklearn.ensemble import BaggingClassifier


model = BaggingClassifier(n_estimators=50, n_jobs=3)
model.fit(x_train, y_train)

predictions = model.predict(x_test)
print(accuracy_score(y_test, predictions))
# 34.8% accuracy