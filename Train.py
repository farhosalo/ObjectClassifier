from ImageClassifier import Dataset, Model
import Configuration
import os
import logging
import numpy as np

# Set the environment variable to suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def trainAndSave():
    modelConfig = Configuration.config.get("model")
    datasetConfig = Configuration.config.get("dataset")

    if (modelConfig is None) or (datasetConfig is None):
        raise ValueError("Model or Dataset configuration is missing.")

    datasetDir = datasetConfig.get("DATASET_DIR")
    if datasetDir is None or not isinstance(datasetDir, str):
        datasetDir = "Data"
        logging.warning(
            f"Invalid or missing DATASET_DIR in configuration. Using default dataset directory {datasetDir}."
        )
    if not os.path.exists(datasetDir):
        raise ValueError(f"Dataset directory {datasetDir} does not exist.")

    batchSize = datasetConfig.get("BATCH_SIZE")
    if batchSize is None or not isinstance(batchSize, int) or batchSize <= 0:
        batchSize = 10
        logging.warning(
            f"Invalid or missing BATCH_SIZE in configuration. Using default batch size of {batchSize}."
        )

    classNamesFile = datasetConfig.get("CLASS_NAME_FILE")
    if classNamesFile is None or not isinstance(classNamesFile, str):
        classNamesFile = "ClassName.txt"
        logging.warning(
            f"Invalid or missing CLASS_NAME_FILE in configuration. Using default {classNamesFile}."
        )

    inputHeight = datasetConfig.get("IMAGE_HEIGHT")
    if inputHeight is None or not isinstance(inputHeight, int) or inputHeight <= 0:
        inputHeight = 24
        logging.warning(
            f"Invalid or missing IMAGE_HEIGHT in configuration, using default {inputHeight}."
        )

    inputWidth = datasetConfig.get("IMAGE_WIDTH")
    if inputWidth is None or not isinstance(inputWidth, int) or inputWidth <= 0:
        inputWidth = 24
        logging.warning(
            f"Invalid or missing IMAGE_WIDTH in configuration, using default {inputWidth}."
        )

    numberOfEpochs = modelConfig.get("EPOCHS")
    if (
        numberOfEpochs is None
        or not isinstance(numberOfEpochs, int)
        or numberOfEpochs <= 0
    ):
        numberOfEpochs = 20
        logging.warning(
            f"Invalid or missing EPOCHS in configuration, using default {numberOfEpochs}."
        )
    modelPath = modelConfig.get("MODEL_PATH")
    if modelPath is None or not isinstance(modelPath, str):
        modelPath = "ImageClassifier.keras"
        logging.warning(
            f"Invalid or missing MODEL_PATH in configuration, using default {modelPath}."
        )

    imageDataset = Dataset.Dataset(
        datasetDir=datasetDir,
        imageWidth=inputWidth,
        imageHeight=inputHeight,
        batchSize=batchSize,
    )

    # Get class names from dataset
    classNames = imageDataset.getClassNames()
    nClasses = len(classNames)

    np.savetxt(classNamesFile, classNames, fmt="%s")

    trainData, validData, testData = imageDataset.getData(
        trainRatio=0.8, validRatio=0.2
    )

    logging.info(f"\nTrain data size={len(list(trainData)) * batchSize}")
    logging.info(f"test data size={len(list(testData)) * batchSize}")
    logging.info(f"and validation data size={len(list(validData)) * batchSize}")
    logging.info(f"Number of classes = {nClasses}\n")

    imageDataset.plotClassDistribution()
    imageDataset.plotExamplesFromDataset(7)

    model = Model.Model(
        inputWidth=inputWidth,
        inputHeight=inputHeight,
        nClasses=nClasses,
    )
    model.plotModel()
    model.train(
        nEpochs=numberOfEpochs,
        trainDataset=trainData,
        validDataset=validData,
        testDataset=testData,
    )
    model.plotTrainingHistory()
    model.saveModel(filePath=modelPath)


if __name__ == "__main__":
    trainAndSave()
