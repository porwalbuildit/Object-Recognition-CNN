# Object Recognition in Images using CNN

## Project Overview

This project implements an image recognition system using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset. The model classifies images into one of ten categories.

---

## Dataset

- Dataset: CIFAR-10
- Total Images: 60,000
- Training Images: 50,000
- Testing Images: 10,000
- Image Size: 32×32 RGB
- Classes:
  - Airplane
  - Automobile
  - Bird
  - Cat
  - Deer
  - Dog
  - Frog
  - Horse
  - Ship
  - Truck

---

## Project Structure

```
Object_Recognition_CNN/

config.py
utils.py
model.py
train.py
predict.py

models/
plots/
images/
```

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn

---

## CNN Architecture

- Convolution Layers
- Batch Normalization
- MaxPooling
- Dropout
- Dense Layers
- Softmax Output

---

## Model Performance

Test Accuracy: ______%

Fill this after training.

---

## How to Run

Install dependencies

```
pip install -r requirements.txt
```

Train Model

```
python train.py
```

Predict Custom Image

```
python predict.py
```

---

## Output

- Trained Model
- Accuracy Graph
- Loss Graph
- Confusion Matrix
- Classification Report

---

## Author

Atharv Porwal