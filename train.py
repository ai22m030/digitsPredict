from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from joblib import dump

digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape(n_samples, -1)

X_train, X_test, y_train, y_test = train_test_split(data, digits.target)
clf = svm.SVC()

clf.fit(X_train, y_train)

dump(clf, 'digits.joblib')
