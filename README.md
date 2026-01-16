# Handwritten Digit Recognition using PyTorch and Pygame

## Overview

This project implements a **handwritten digit recognition system** using a fully connected Convolutional Neural Network (CNN) trained on the **MNIST dataset** with **PyTorch**.
After training, the model is integrated with a **Pygame drawing interface**, allowing users to draw digits on the screen and receive real-time predictions from the trained model.

---

## Features

* Trains a neural network on the MNIST handwritten digit dataset
* Achieves classification of digits from 0 to 9
* Interactive drawing window using Pygame
* Automatic preprocessing of drawn digits (grayscale, cropping, resizing, normalization)
* Real-time digit prediction

---

## Technologies Used

* Python 3
* PyTorch
* Torchvision
* NumPy
* OpenCV (cv2)
* Pygame

---

## Dataset

* **MNIST Dataset**
* 60,000 training images
* 10,000 testing images
* Image size: 28 × 28 grayscale

Dataset is automatically downloaded using `torchvision.datasets.MNIST`.

---

## Installation

Install required libraries:

```bash
pip install torch torchvision pygame numpy opencv-python
```

---

## How to Run

1. Save the file (for example):
   `digit_recognition.py`

2. Run the program:

```bash
python digit_recognition.py
```

3. The program will:

   * Train the neural network
   * Test it on the MNIST test set
   * Display model accuracy
   * Open a drawing window

---

## How to Use the Drawing Window

* **Hold left mouse button** → Draw a digit
* **Release mouse button** → Model predicts the digit
* **Press `C`** → Clear the screen
* **Close window** → Exit program

The predicted digit will be displayed at the bottom of the window.

---

## Image Processing Pipeline

The drawn digit is processed as follows:

1. Convert Pygame surface to NumPy array
2. Convert to grayscale
3. Apply binary threshold
4. Crop digit using bounding box
5. Resize to 20 × 20
6. Pad to 28 × 28
7. Normalize using MNIST mean and standard deviation
8. Convert to PyTorch tensor

This ensures compatibility with the MNIST-trained model.
