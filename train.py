from keras.datasets import mnist
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from joblib import dump
import matplotlib.pyplot as plt
from numpy import asarray
import numpy as np
from PIL import Image
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import imageio.v2 as imageio
from matplotlib import pyplot as plt


num_classes = 10
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float64')
x_test = x_test.astype('float64')

'''
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
batch_size = 128
epochs = 2

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save("test_model.h5")
'''
im = imageio.imread('fyPhv.jpg')

# normalize image
gray = np.dot(im[..., :3], [0.299, 0.587, 0.114])

# im = im.astype('float64')

# gray = asarray(im)

# plt.imshow(gray, cmap=plt.get_cmap('gray'))
# plt.show()

plt.imshow(x_test[3], cmap=plt.get_cmap('gray'))
plt.show()

# gray /= 255


model = load_model("test_model.h5")
prediction = model.predict(x_test)
print(prediction.argmax())
print(prediction)

'''
image_index = 35
print(y_train[image_index])
plt.imshow(x_train[image_index], cmap='Greys')
plt.show()

digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape(n_samples, -1)

X_train, X_test, y_train, y_test = train_test_split(data, digits.target)
clf = svm.SVC()

clf.fit(X_train, y_train)
print(X_train)
dump(clf, 'digits.joblib')
'''
