###################### Import
from sklearn.neural_network import MLPClassifier
from preprocess_01 import *
####################### Red neuronal con 2 capas de 5 neuronas


num_folds = 10
n_estimators = 100
num_instances = len(x_train)
seed = 7
scoring = 'accuracy'

model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                      hidden_layer_sizes=(5, 2), random_state=1)

model.fit(x_train_scaled, y_train)
predictions =  model.predict(x_test_scaled)


print "*"*30
print  "%s" % (accuracy_score(y_test, predictions))
print "*"*30
#Accuracy del 28%