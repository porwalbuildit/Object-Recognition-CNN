# ==========================================
# Utility Functions
# ==========================================

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix

from config import *


# ------------------------------------------
# Create Required Folders
# ------------------------------------------

def create_folders():
    """
    Create project folders if they don't exist.
    """

    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("images", exist_ok=True)


# ------------------------------------------
# Set Random Seed
# ------------------------------------------

def set_seed():
    """
    Set random seed for reproducibility.
    """

    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


# ------------------------------------------
# Load Dataset
# ------------------------------------------

def load_dataset():

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES)

    return X_train, y_train, y_train_cat, X_test, y_test, y_test_cat


# ------------------------------------------
# Show Sample Images
# ------------------------------------------

def show_samples(images, labels):

    plt.figure(figsize=(12,6))

    for i in range(10):

        plt.subplot(2,5,i+1)

        plt.imshow(images[i])

        plt.title(CLASS_NAMES[int(labels[i])])

        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ------------------------------------------
# Data Augmentation
# ------------------------------------------

def get_data_generator():

    datagen = ImageDataGenerator(

        rotation_range=15,

        width_shift_range=0.1,

        height_shift_range=0.1,

        horizontal_flip=True,

        zoom_range=0.2

    )

    return datagen


# ------------------------------------------
# Accuracy Plot
# ------------------------------------------

def plot_accuracy(history):

    plt.figure(figsize=(8,5))

    plt.plot(history.history["accuracy"], label="Training")

    plt.plot(history.history["val_accuracy"], label="Validation")

    plt.title("Model Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.savefig(ACCURACY_PLOT)

    plt.show()


# ------------------------------------------
# Loss Plot
# ------------------------------------------

def plot_loss(history):

    plt.figure(figsize=(8,5))

    plt.plot(history.history["loss"], label="Training")

    plt.plot(history.history["val_loss"], label="Validation")

    plt.title("Model Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.savefig(LOSS_PLOT)

    plt.show()


# ------------------------------------------
# Confusion Matrix
# ------------------------------------------

def plot_confusion_matrix(model, X_test, y_test):

    predictions = model.predict(X_test)

    predicted_classes = np.argmax(predictions, axis=1)

    true_classes = y_test.flatten()

    cm = confusion_matrix(true_classes, predicted_classes)

    plt.figure(figsize=(10,8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(CONFUSION_MATRIX)

    plt.show()