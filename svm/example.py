import matplotlib.pyplot as plt
from sklearn import datasets, svm

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.001, C=100)

x,y = digits.data[:-1], digits.target[:-1]

clf.fit(x,y)

print(clf.predict(digits.data[-4].reshape(1, -1)))

plt.imshow(digits.images[-4], cmap=plt.cm.gray_r, interpolation="nearest")

plt.show()