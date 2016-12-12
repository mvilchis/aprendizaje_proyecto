__author__ = 'tim'
from sklearn.ensemble import *

clf1 = BaggingClassifier(n_estimators=50, n_jobs=3)
clf2 = RandomForestClassifier(n_jobs=2, n_estimators=10)
clf3 = AdaBoostClassifier(n_estimators=100)
model = VotingClassifier(estimators=[ ('bc', clf1), ('rf', clf2), ('abc', clf3)], weights=[2,2,1])

model.fit(x_train, y_train)

predictions = model.predict(x_test)
print(accuracy_score(y_test, predictions))
# accuracy 34.8%