import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd

# breast-cancer-wisconsin data from:
# https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.names

# reading data
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
# ? is for unknow attributes
df.replace('?', -99999, inplace=True)
# id is not relevant
df.drop(['id'], 1, inplace=True)

# features => everything less class
X = np.array(df.drop(['class'], 1))
# labels => 2 = benign, 4 = malignant
y = np.array(df['class'])

# traning data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
# clf = neighbors.KNeighborsClassifier()
clf = svm.SVC()
clf.fit(X_train, y_train)

# testing
accuracy = clf.score(X_test, y_test)
print('Accuracy: ',accuracy)

# sample data
exemple_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,4,1,2,3,2,1]])
exemple_measures = exemple_measures.reshape(len(exemple_measures),-1)

prediction = clf.predict(exemple_measures)
print('Prediction: ',prediction)