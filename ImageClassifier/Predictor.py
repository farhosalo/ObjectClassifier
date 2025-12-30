import tensorflow as tf
import os
import numpy as np
from PIL import Image
import logging


class Predictor:
    def __init__(self, modelPath=None, classNamePath=None, height=224, width=224):
        self.__Model__ = None
        self.__ClassName__ = None
        if modelPath is None or classNamePath is None:
            raise ValueError("Both model path and class name path must be provided.")
        self.__loadModel__(modelPath)
        self.__loadClassNames__(classNamePath)

        # Create a preprocessing pipeline
        self.__Preprocess__ = tf.keras.Sequential(
            [
                tf.keras.layers.Resizing(
                    height=height, width=width, crop_to_aspect_ratio=True
                ),
            ]
        )

    def __loadModel__(self, modelPath=None):
        # check if the file exists
        if not os.path.isfile(modelPath):
            raise FileNotFoundError(f"Model file {modelPath} does not exist.")

        # Load the model
        self.__Model__ = tf.keras.models.load_model(modelPath)
        logging.info(f"Model loaded from {modelPath}")

    def __loadClassNames__(self, classNamePath=None):
        # check if the file exists
        if not os.path.isfile(classNamePath):
            raise FileNotFoundError(f"Class names file {classNamePath} does not exist.")

        self.__ClassName__ = np.loadtxt(classNamePath, dtype=str)

    def predict(self, image):
        if not os.path.isfile(image):
            raise FileNotFoundError(f"Image file {image} does not exist.")

        img = Image.open(image)
        img = np.array(img)
        # preProcessedImage = testDatase.map(lambda X, Y: (common.preprocess(X), Y))
        preProcessedImage = self.__Preprocess__(img)
        prediction = self.__Model__.predict(preProcessedImage[None, :, :])
        if prediction.max() > 0.95:
            yProba = np.argmax(prediction)
            return self.__ClassName__[yProba]
        return "Unknown"
