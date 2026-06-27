# ==========================================
# Predict Custom Image using Trained CNN
# ==========================================

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from config import *

# ------------------------------------------
# Load Trained Model
# ------------------------------------------

print("Loading trained model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!\n")

# ------------------------------------------
# Image Path
# ------------------------------------------

IMAGE_PATH = "images/test.jpg"

# ------------------------------------------
# Load Image
# ------------------------------------------

try:
    # Original image (for display)
    original_img = image.load_img(IMAGE_PATH)

    # Resized image (for prediction)
    img = image.load_img(
        IMAGE_PATH,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
    )

except FileNotFoundError:
    print(f"Image not found: {IMAGE_PATH}")
    print("Place an image named 'test.jpg' inside the images folder.")
    exit()

# Convert image to array
img_array = image.img_to_array(img)

# Normalize
img_array = img_array.astype("float32") / 255.0

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# ------------------------------------------
# Prediction
# ------------------------------------------

prediction = model.predict(img_array, verbose=0)[0]

predicted_index = np.argmax(prediction)
confidence = prediction[predicted_index] * 100

# Get Top 3 Predictions
top3 = prediction.argsort()[-3:][::-1]

# ------------------------------------------
# Display Image
# ------------------------------------------

plt.figure(figsize=(6,6))

plt.imshow(original_img)

plt.axis("off")

plt.title(
    f"Prediction: {CLASS_NAMES[predicted_index]}\nConfidence: {confidence:.2f}%"
)

plt.show()

# ------------------------------------------
# Print Results
# ------------------------------------------

print("=" * 50)
print("Prediction Result")
print("=" * 50)

print(f"Predicted Class : {CLASS_NAMES[predicted_index]}")
print(f"Confidence      : {confidence:.2f}%")

print("\nTop 3 Predictions")
print("-" * 30)

for i in top3:
    print(f"{CLASS_NAMES[i]:12} : {prediction[i] * 100:.2f}%")

print("=" * 50)