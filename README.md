# Convolutional Neural Network (CNN) on MNIST Dataset using NumPy

Note! This is an educational example demonstrating low performance. The true goal is an in-depth study of the mathematical foundations of convolutional neural networks.

This repository contains an implementation of a Convolutional Neural Network (CNN) for the MNIST dataset, written entirely using NumPy. The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9).


## Introduction
This project demonstrates the implementation of a basic Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The model is implemented using only NumPy to give a clear understanding of the underlying operations in a CNN.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/erastov-alex/cnn-numpy.git
   cd MNIST_CNN_Numpy
   ```
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download MNIST csv dataset 

Implementation Details
Data Preparation
The data is read from CSV files where each row corresponds to a flattened 28x28 image. The first column contains the label, and the remaining columns contain the pixel values (0-255).

## Model Architecture
The CNN consists of the following layers:

- Convolutional Layer: Applies a set of 3x3 kernels to the input image.
- ReLU Activation: Applies the ReLU function element-wise.
- Max Pooling Layer: Reduces the dimensionality by taking the maximum value over a 2x2 window.
- Dense Layer: Fully connected layer for classification.

## Training
The training loop performs the following steps for each epoch:

- Forward propagation through the network.
- Calculation of the loss using the cross-entropy loss function.
- Backward propagation to compute gradients.
- Update the weights using Stochastic Gradient Descent (SGD).

## Testing
After training, the model's accuracy is evaluated on the test set.

## Results
After training for a few epochs, the model achieves a reasonable accuracy on the test set. The training and evaluation details are printed to the console during execution.