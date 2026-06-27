# ==========================================
# CNN Model Architecture
# ==========================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense,
    Input
)
from tensorflow.keras.optimizers import Adam

from config import *


def build_model():

    model = Sequential([

        # Input Layer
        Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)),

        # =========================
        # Block 1
        # =========================
        Conv2D(32, (3,3), padding="same", activation="relu"),
        BatchNormalization(),

        Conv2D(32, (3,3), activation="relu"),
        BatchNormalization(),

        MaxPooling2D((2,2)),
        Dropout(0.25),

        # =========================
        # Block 2
        # =========================
        Conv2D(64, (3,3), padding="same", activation="relu"),
        BatchNormalization(),

        Conv2D(64, (3,3), activation="relu"),
        BatchNormalization(),

        MaxPooling2D((2,2)),
        Dropout(0.30),

        # =========================
        # Block 3
        # =========================
        Conv2D(128, (3,3), padding="same", activation="relu"),
        BatchNormalization(),

        Conv2D(128, (3,3), activation="relu"),
        BatchNormalization(),

        MaxPooling2D((2,2)),
        Dropout(0.40),

        # =========================
        # Fully Connected Layers
        # =========================
        Flatten(),

        Dense(512, activation="relu"),
        Dropout(0.5),

        Dense(NUM_CLASSES, activation="softmax")

    ])

    optimizer = Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model