import skflow
from sklearn.metrics import accuracy_score

classifier = skflow.TensorFlowDNNClassifier(hidden_units=[100, 1000, 100], n_classes=3)
classifier.fit(x_train, y_train)

predictions = classifier.predict(x_test)
print(accuracy_score(y_test, predictions2))
# accuracy del 33%