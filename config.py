# ==========================================
# Project Configuration
# ==========================================

# Image Settings
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_CHANNELS = 3

# Dataset
NUM_CLASSES = 10

CLASS_NAMES = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

# Training
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001

# Paths
MODEL_PATH = "models/cifar10_cnn_model.keras"

ACCURACY_PLOT = "plots/accuracy.png"
LOSS_PLOT = "plots/loss.png"
CONFUSION_MATRIX = "plots/confusion_matrix.png"

# Random Seed
RANDOM_SEED = 42