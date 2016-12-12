__author__ = 'tim'
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(verbose=1)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print(accuracy_score(y_test, predictions))
# 35% accuracy 30 min entrenamiento