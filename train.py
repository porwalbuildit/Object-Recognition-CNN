# ==========================================
# Train CNN on CIFAR-10
# ==========================================

import os
import numpy as np
from sklearn.metrics import classification_report

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from config import *
from utils import (
    create_folders,
    set_seed,
    load_dataset,
    show_samples,
    get_data_generator,
    plot_accuracy,
    plot_loss,
    plot_confusion_matrix
)
from model import build_model


def main():

    # --------------------------------------
    # Setup
    # --------------------------------------
    create_folders()
    set_seed()

    # --------------------------------------
    # Load Dataset
    # --------------------------------------
    print("=" * 50)
    print("Loading CIFAR-10 Dataset...")
    print("=" * 50)

    X_train, y_train, y_train_cat, X_test, y_test, y_test_cat = load_dataset()

    print(f"Training Images : {X_train.shape}")
    print(f"Testing Images  : {X_test.shape}")

    # Display sample images
    show_samples(X_train, y_train)

    # --------------------------------------
    # Data Augmentation
    # --------------------------------------
    datagen = get_data_generator()
    datagen.fit(X_train)

    # --------------------------------------
    # Build Model
    # --------------------------------------
    print("\nBuilding CNN Model...\n")

    model = build_model()

    model.summary()

    # --------------------------------------
    # Callbacks
    # --------------------------------------

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    # --------------------------------------
    # Training
    # --------------------------------------

    print("\nStarting Training...\n")

    history = model.fit(
        datagen.flow(
            X_train,
            y_train_cat,
            batch_size=BATCH_SIZE
        ),
        epochs=EPOCHS,
        validation_data=(X_test, y_test_cat),
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )

    # --------------------------------------
    # Evaluation
    # --------------------------------------

    print("\nEvaluating Model...\n")

    loss, accuracy = model.evaluate(
        X_test,
        y_test_cat,
        verbose=1
    )

    print("=" * 50)
    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"Test Loss     : {loss:.4f}")
    print("=" * 50)

    # --------------------------------------
    # Graphs
    # --------------------------------------

    plot_accuracy(history)

    plot_loss(history)

    # --------------------------------------
    # Classification Report
    # --------------------------------------

    predictions = model.predict(X_test)

    predicted_classes = np.argmax(predictions, axis=1)

    true_classes = y_test.flatten()

    print("\nClassification Report\n")

    print(classification_report(
        true_classes,
        predicted_classes,
        target_names=CLASS_NAMES
    ))

    # --------------------------------------
    # Confusion Matrix
    # --------------------------------------

    plot_confusion_matrix(model, X_test, y_test)

    print("\nTraining Completed Successfully!")

    print(f"\nModel saved to:\n{MODEL_PATH}")


if __name__ == "__main__":
    main()