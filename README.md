# Image Classifier

This project aims to create a robust image classifier using a relatively small dataset. We’ll use TensorFlow and transfer learning, specifically the „Xception“ model and the „imagenet“ dataset.

The project consists of two main parts:

- Model Training: The first phase involves training the model using the provided dataset.
- Model Utilization: The second phase involves using the trained model to classify images.

## Prepare environment

This project relies on TensorFlow and other Python third-party libraries listed in the requirement.txt file. These libraries must be installed before you can start using the project.

To make this easier, I recommend using a Python virtual environment. This will create a clean and isolated environment that won’t interfere with your existing one. To create a virtual environment and install all dependencies, run the following command from your terminal:

``` Shell
conda create -n ImageClassifier python=3.12
conda activate ImageClassifier
pip install -r requirements.txt
```

## Install the project

To install the project, run the following two commands from your terminal:

```Shell
git clone https://github.com/farhosalo/ImageClassifier.git
cd ImageClassifier
```

## Configuration (For Expert Users)

Please don't modifying the configuration unless you have extensive experience in creating such models.

The configuration is stored in a Python file named Configuration.py located in the project root directory.

The configuration consists of two sections:

- **Dataset Section:**
  - **DATASET_DIR:** Directory where training images are saved.
  - **IMAGE_HEIGHT:** Training image height in pixels.
  - **IMAGE_WIDTH:** Training image width in pixels.
  - **BATCH_SIZE:** The number of images used in one learning iteration.
  - **CLASS_NAME_FILE:** The file where class names will be saved.

**Note:** The „IMAGE_HEIGHT“ and „IMAGE_WIDTH“ represent the dimensions to which images will be resized before training. Your training images may be smaller or larger than these dimensions.

- **Model Section:**
  - **EPOCHS:** Number of complete iterations of the learning process across the entire training dataset
  - **MODEL_PATH:** Path to where the model will be saved.

## Training and saving the model

You have to organize the training images in a directory structure like this: <br/><br/>

``` Shell
    Data
    |--Class1
    |  |--filename1.jpg
    |  |--filename2.jpg
    |  |--filename3.jpg
    |  |--...
    |--Class2
    |  |--filename1.jpg
    |  |--filename2.jpg
    |  |--filename3.jpg
    |  |-- ...
    |..................
```

The subdirectories within the Data directory represent classes. The class names are the names of the subfolders. File names are irrelevant and can be arbitrarily assigned.

To start training, run the following command from your terminal:

```Shell
python Train.py
```

The training process will take a few minutes, depending on your hardware and the size of the training data. Once the training is complete, the model will be saved by default in the project's root directory as a file named „ImageClassifier.keras“. Class names will be saved by default in a file named „ClassName.txt“ in the project’s root directory. This file is used later for predicting class names instead of class indexes. It is recommended that this file not be modified manually.

## Predicting image classes

After training and saving the model, we reload it and use it to predict image classes.
To do this, run the following command in your terminal:

```Shell
python Predict.py -i <path to image>
```

This command will output the predicted class of the input image you provide.

## Contributing

Contributions are welcome! If you find any bugs or have ideas for new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
