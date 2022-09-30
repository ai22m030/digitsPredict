import imageio.v2 as imageio
import numpy as np
from keras.models import load_model
import cv2


def predict(filename, model_name, console_output=True):
    # Load and preprocess image (resize, grayscale)
    image = imageio.imread(filename)
    image = cv2.resize(image, (28, 28))
    gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    gray = gray.reshape(1, 28, 28, 1)
    gray /= 255

    # Predict digit based on loaded model
    model = load_model(model_name)
    prediction = model.predict(gray)
    probability_for_detected_class = round(100 * prediction[0][prediction.argmax()], 2)

    if console_output:
        print(".-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-.")
        print(f"\nThe given image with filename '{filename}' was predicted as: {prediction.argmax()} ({probability_for_detected_class}%).")
        print(f"\nProbability for all detectable classes:")
        for index, probability in enumerate(np.nditer(prediction)):
            print(f"{index}: {round(float(probability) * 100, 2)}%")
        print(".-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-.")

    return prediction.argmax()
