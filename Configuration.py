config = {
    "dataset": {  # Dataset configuration
        "DATASET_DIR": "Data",  # Path to the dataset directory
        "IMAGE_HEIGHT": 224,  # Input Height
        "IMAGE_WIDTH": 224,  # Input Width
        "BATCH_SIZE": 10,  # Batch Size for training and validation
        "CLASS_NAME_FILE": "ClassName.txt",  # File to save class names
    },
    "model": {  # Model configuration
        "EPOCHS": 20,  # Number of training epochs
        "MODEL_PATH": "./ImageClassifier.keras",  # Path to save or load the model
    },
}
