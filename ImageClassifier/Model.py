import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
import math
import logging

from tensorflow.keras.applications.xception import Xception as BaseModel
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input


class Model:
    def __init__(self, inputWidth, inputHeight, nClasses):
        self.__InputShape = (inputWidth, inputHeight) + (3,)
        self.__BaseModel__ = BaseModel(
            weights="imagenet", include_top=False, input_shape=self.__InputShape
        )
        self.__NClasses__ = nClasses
        learningRate = 0.01

        self.__TrainingPhases__ = {
            "FEATURE_EXTRACTION": {
                "learningRate": learningRate,
                "freezeBaseModel": True,
                "trainingHistory": None,
                "nEpochsFactor": 0.67,
            },
            "FINE_TUNING": {
                "learningRate": learningRate / 100,
                "freezeBaseModel": False,
                "trainingHistory": None,
                "nEpochsFactor": 0.33,
            },
        }

        self.__createModel__()

    def __createModel__(self):
        inputs = tf.keras.layers.Input(shape=self.__InputShape, name="RSM_Input")
        x = self.__createAugmentedLayers__()(inputs)
        x = preprocess_input(x)
        x = self.__BaseModel__(
            x,
            # To make sure that the base model is running in inference mode, so that batchnorm
            # statistics don't get updated even after we unfreeze the base model for fine-tuning.
            training=False,
        )
        x = tf.keras.layers.GlobalAveragePooling2D(name="RSM_GlobalAvgPooling2D")(x)

        # Regularize with dropout
        x = tf.keras.layers.Dropout(0.2, name="RSM_Dropout")(x)
        outputs = tf.keras.layers.Dense(
            self.__NClasses__, activation="softmax", name="RSM_Output"
        )(x)

        # Create a new model
        self.__Model__ = tf.keras.Model(inputs=inputs, outputs=outputs)

    def __createAugmentedLayers__(self):
        randomSeed = 42
        return tf.keras.Sequential(
            [
                tf.keras.layers.RandomRotation(factor=(-0.02, 0.02), seed=randomSeed),
                tf.keras.layers.RandomContrast(factor=0.1, seed=randomSeed),
            ],
            name="RSM_DataAugmentation",
        )

    def plotModel(self, filePath="./model.png"):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filePath), exist_ok=True)
        # Plot the model architecture and save it to a file
        tf.keras.utils.plot_model(
            model=self.__Model__, to_file=filePath, show_shapes=True
        )

    def train(self, nEpochs, trainDataset, validDataset, testDataset=None):
        if len(trainDataset) == 0 or len(validDataset) == 0:
            raise ValueError("Train or validation dataset is empty.")

        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        momentum = 0.9

        for phase, config in self.__TrainingPhases__.items():
            logging.info(f"\nTraining phase: {phase}")

            if config["freezeBaseModel"]:
                self.__BaseModel__.trainable = False
            else:
                self.__BaseModel__.trainable = True
                finetuneAt = (2 * len(self.__BaseModel__.layers)) // 3
                logging.info(f"Fine tune from layer {finetuneAt}")
                for layer in self.__BaseModel__.layers[:finetuneAt]:
                    layer.trainable = False

            optimizer = tf.keras.optimizers.SGD(
                learning_rate=config["learningRate"], momentum=momentum
            )
            self.__Model__.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

            self.__Model__.summary()

            # Training
            start = time.time()

            config["trainingHistory"] = self.__Model__.fit(
                trainDataset,
                validation_data=validDataset,
                epochs=math.ceil(nEpochs * config["nEpochsFactor"]),
                verbose=True,
            )
            stop = time.time()
            learningDuration = (stop - start) / 60

            logging.info(f"{phase} learning phase took: {learningDuration} minutes")

            if len(testDataset) != 0:
                logging.info("Evaluate the model on test data")
                self.__Model__.evaluate(testDataset)

    def plotTrainingHistory(self):
        index = 0
        nRows = len(self.__TrainingPhases__)
        for phase, config in self.__TrainingPhases__.items():
            if config["trainingHistory"] is None:
                logging.warn(f"No training history for {phase} phase available.")
                nRows -= 1
                continue
            for metric in ["accuracy", "loss"]:
                index += 1
                plt.subplot(nRows, 2, index)
                plt.plot(config["trainingHistory"].history[metric])
                plt.plot(config["trainingHistory"].history["val_" + metric])
                plt.title(phase)
                plt.ylabel(metric)
                plt.xlabel("epoch")
                plt.legend(["train", "val"])

        if nRows != 0:
            plt.tight_layout()
            plt.show()

    def saveModel(self, filePath):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filePath), exist_ok=True)
        # Save the model
        self.__Model__.save(filePath)
        logging.info(f"Model saved to {filePath}")
