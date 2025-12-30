from ImageClassifier import Predictor
import Configuration
import os
import shutil
import logging
import argparse
import filetype

# Set the environment variable to suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def isImage(filename: str):
    if filename is None or not os.path.isfile(filename):
        return False
    mime = filetype.guess_mime(filename)
    if mime is None:
        return False
    return mime.startswith("image/")


def loadAndPredict():
    parser = argparse.ArgumentParser(
        prog="ImageClassifier",
        description="Predicts the class of an image passed as input.",
    )

    parser.add_argument("-i", "--input", required=True, help="Path to the input image")

    args = parser.parse_args()
    if not os.path.exists(args.input):
        logging.fatal("Input path does not exist")
        return
    if not isImage(args.input):
        logging.fatal("Input file is not an image")
        return

    modelConfig = Configuration.config["model"]
    datasetConfig = Configuration.config["dataset"]

    imagePredictor = Predictor.Predictor(
        modelPath=modelConfig["MODEL_PATH"],
        classNamePath=datasetConfig["CLASS_NAME_FILE"],
    )

    pred = imagePredictor.predict(args.input)
    print(pred)


if __name__ == "__main__":
    loadAndPredict()
