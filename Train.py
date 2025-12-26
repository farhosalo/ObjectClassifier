from ObjectClassifier import ObjectDataset
import Configuration
import os
import numpy as np
from PIL import Image

# Set the environment variable to suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def trainAndSave():
    appConfig = Configuration.config["app"]
    modelConfig = Configuration.config["model"]
    datasetConfig = Configuration.config["dataset"]

    batchSize = datasetConfig["BATCH_SIZE"]
    roadSignDataset = ObjectDataset.ObjectDataset(
        datasetDir=datasetConfig["DATASET_DIR"],
        imageWidth=datasetConfig["IMAGE_WIDTH"],
        imageHeight=datasetConfig["IMAGE_HEIGHT"],
        batchSize=datasetConfig["BATCH_SIZE"],
    )

    # Get class names from dataset
    classNames = roadSignDataset.getClassNames()
    nClasses = len(classNames)

    np.savetxt(appConfig["CLASS_NAME_FILE"], classNames, fmt="%s")

    trainData, validData, testData = roadSignDataset.getData(
        trainRatio=0.8, validRatio=0.2
    )

    print(f"\nTrain data size={len(list(trainData)) * batchSize}")
    print(f"test data size={len(list(testData)) * batchSize}")
    print(f"and validation data size={len(list(validData)) * batchSize}")
    print(f"Number of classes = {nClasses}\n")

    roadSignDataset.plotClassDistribution()
    roadSignDataset.plotExamplesFromDataset(7)


if __name__ == "__main__":
    trainAndSave()
