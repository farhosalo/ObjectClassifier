import tensorflow as tf
import os.path
import matplotlib.pyplot as plt
import seaborn as sns
import logging


class Dataset:
    def __init__(self, datasetDir, imageWidth, imageHeight, batchSize):
        self.__DatasetDir = datasetDir
        self.__ImageWidth = imageWidth
        self.__ImageHeight = imageHeight
        self.__BatchSize = batchSize
        self.__autotune = tf.data.AUTOTUNE

        self.__LearnData = self.__loadData()

    def __loadData(self):
        return tf.keras.utils.image_dataset_from_directory(
            directory=self.__DatasetDir,
            shuffle=True,
            batch_size=None,
            image_size=(self.__ImageHeight, self.__ImageWidth),
        )

    def getClassNames(self):
        return self.__LearnData.class_names

    def getData(self, trainRatio=0.7, validRatio=0.2):
        if (
            trainRatio <= 0
            or trainRatio >= 1
            or validRatio >= 1
            or validRatio < 0
            or trainRatio + validRatio > 1
        ):
            raise ValueError("trainRatio or validRatio no valid")

        # Calculate the sizes of the training, validation, and testing sets
        # based on the specified ratios.
        learnDataLength = len(list(self.__LearnData))

        trainSize = int(learnDataLength * trainRatio)
        validSize = int(learnDataLength * validRatio)
        testSize = learnDataLength - trainSize - validSize

        trainData = self.__LearnData.take(trainSize)
        validData = self.__LearnData.skip(trainSize).take(validSize)

        testData = self.__LearnData.skip(trainSize + validSize).take(testSize)

        # Do buffered prefetching to avoid i/o blocking
        trainData = trainData.batch(self.__BatchSize).prefetch(
            buffer_size=self.__autotune
        )
        testData = testData.batch(self.__BatchSize).prefetch(
            buffer_size=self.__autotune
        )
        validData = validData.batch(self.__BatchSize).prefetch(
            buffer_size=self.__autotune
        )

        return trainData, validData, testData

    def __createExample(self, image, label):
        image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        image = tf.io.encode_jpeg(image)

        feature = {
            "images": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image.numpy()])
            ),
            "labels": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))

    def __parseRecord(self, example):
        feature_description = {
            "images": tf.io.FixedLenFeature([], tf.string),
            "labels": tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(example, feature_description)
        example["images"] = tf.image.convert_image_dtype(
            tf.io.decode_jpeg(example["images"], channels=3), dtype=tf.float32
        )

        return example["images"], example["labels"]

    def save2Records(self, path):
        if not os.path.isfile(path):
            with tf.io.TFRecordWriter(path) as writer:
                for image, label in self.__LearnData:
                    tf_example = self.__createExample(image, label)
                    writer.write(tf_example.SerializeToString())
        else:
            logging.info("File already exists")

    def getDataFromRecords(self, path, trainRatio=0.7, validRatio=0.2):
        if not os.path.isfile(path):
            raise OSError("File does not exist")

        # Create a dataset from the TFRecord file.
        rawDataset = tf.data.TFRecordDataset(path)
        dataset = rawDataset.map(self.__parseRecord)

        trainData, validData, testData = self.split(dataset, trainRatio, validRatio)

        # Do buffered prefetching to avoid i/o blocking
        trainData = trainData.prefetch(buffer_size=self.__autotune)
        testData = testData.prefetch(buffer_size=self.__autotune)
        validData = validData.prefetch(buffer_size=self.__autotune)

        # prefetching to improve performance
        trainData = trainData.prefetch(buffer_size=self.__autotune)
        testData = testData.prefetch(buffer_size=self.__autotune)
        validData = validData.prefetch(buffer_size=self.__autotune)

        return trainData, validData, testData

    def plotClassDistribution(self):
        plt.figure(constrained_layout=True)
        classNames = self.getClassNames()
        labelCount = [classNames[label.numpy()] for image, label in self.__LearnData]
        sns.countplot(labelCount)
        plt.tight_layout()
        plt.show()

    def plotExamplesFromDataset(self, count=5):
        classNames = self.getClassNames()
        plt.figure(constrained_layout=True)
        nCols = min(count, 5)
        nRows = (count // 5) + 1
        for index, (image, label) in enumerate(self.__LearnData.take(count)):
            plt.subplot(nRows, nCols, index + 1)
            plt.imshow((image / 255))
            plt.title(classNames[label.numpy()])
            plt.axis("off")
        plt.tight_layout()
        plt.show()
