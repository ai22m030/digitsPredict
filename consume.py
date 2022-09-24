from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from joblib import load
from PIL import Image
from numpy import asarray

clf = load('digits.joblib')
image = Image.open('fyPhv.jpg').convert('L').resize((64, 64))
data = asarray(image)

print(clf.predict(data))

