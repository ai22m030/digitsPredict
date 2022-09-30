import argparse
import os
from consume_lib import predict


if __name__ == '__main__':
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

    predict(filename, model_name)

