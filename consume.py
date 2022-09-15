from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape(n_samples, -1)

X_train, X_test, y_train, y_test = train_test_split(data, digits.target)
clf = svm.SVC()

clf.fit(X_train, y_train)
predicted = clf.predict(X_test)

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Prediction: %i" % prediction)

plt.show()
