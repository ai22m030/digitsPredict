import imageio.v2 as imageio
import numpy as np
from keras.models import load_model
import argparse
import cv2
import os

# Default value for model
model_name = "assignment01_model.h5"

# Parse arguments
parser = argparse.ArgumentParser(
    prog="Use a previously trained model to predict a handwritten digit.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("-m", "--model", help="Choose a .h5 model that was trained with keras. Default is "
                                          "assignment01_model.h5")
parser.add_argument("-i", "--image", help="Choose an image to predict its class.")
args = parser.parse_args()
if args.model:
    model_name = args.model
else:
    print("Warning: No model provided, looking for default model.")
    if os.path.exists(model_name):
        print(f"Default model detected: {model_name}")
    else:
        exit("Error: No model provided and default model not found. Please provide a model with -m file")
if not args.image:
    exit("Error: No image provided. Please use -i file")
filename = args.image

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

print(".-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-.")
print(f"\nThe given image with filename '{filename}' was predicted as: {prediction.argmax()} ({probability_for_detected_class}%).")
print(f"\nProbability for all detectable classes:")
for index, probability in enumerate(np.nditer(prediction)):
    print(f"{index}: {round(float(probability) * 100, 2)}%")
print(".-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-.")
