import imageio
import numpy as np
#from matplotlib import pyplot as plt
from keras.models import load_model
import cv2

filename = "test_3.jpg" # hard coded filename

im = imageio.imread(filename)
im = cv2.resize(im, (28, 28))
gray = np.dot(im[...,:3], [0.299, 0.587, 0.114])
gray = gray.reshape(1, 28, 28, 1)
gray /= 255
model = load_model("assignment01_model.h5") # hard coded model name

prediction = model.predict(gray)
probability_for_detected_class = round(100*prediction[0][prediction.argmax()], 2)
print(f"\nThe given image with filename '{filename}' was predicted as: {prediction.argmax()} ({probability_for_detected_class} %).")
print(f"\nProbability for all detectable classes (in %):")
for index, probability in enumerate(np.nditer(prediction)):
    print(f"{index}: {round(float(probability)*100, 2)}")
