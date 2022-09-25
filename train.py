import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

mnist_dataset = tf.keras.datasets.mnist

img_rows = img_cols = 28
(x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

model = tf.keras.models.Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
    activation='relu',
    input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer='adam', 
loss="sparse_categorical_crossentropy", 
metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=128, epochs=5, verbose=1, validation_data=(x_test, y_test))
filename = "assignment01_model.h5"
model.save(filename)
print(f"The model was saved as '{filename}' and can be used with consume.py -m {filename}")