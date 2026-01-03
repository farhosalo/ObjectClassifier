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

    modelConfig = Configuration.config.get("model")
    datasetConfig = Configuration.config.get("dataset")

    if (modelConfig is None) or (datasetConfig is None):
        raise ValueError("Model or Dataset configuration is missing.")

    classNamesFile = datasetConfig.get("CLASS_NAME_FILE")
    if classNamesFile is None or not isinstance(classNamesFile, str):
        classNamesFile = "ClassName.txt"
        logging.warning(
            f"Invalid or missing CLASS_NAME_FILE in configuration. Using default {classNamesFile}."
        )

    modelPath = modelConfig.get("MODEL_PATH")
    if modelPath is None or not isinstance(modelPath, str):
        modelPath = "ImageClassifier.keras"
        logging.warning(
            f"Invalid or missing MODEL_PATH in configuration, using default {modelPath}."
        )

    imagePredictor = Predictor.Predictor(
        modelPath=modelPath,
        classNamePath=classNamesFile,
    )

    pred = imagePredictor.predict(args.input)
    print(pred)


if __name__ == "__main__":
    loadAndPredict()
