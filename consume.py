from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from joblib import load

clf = load('digits.joblib')

