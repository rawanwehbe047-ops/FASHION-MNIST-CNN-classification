# Fashion-MNIST Image Classification Using CNN

A beginner-friendly image classification project using **PyTorch** and a **Convolutional Neural Network (CNN)**. The model learns to identify 10 types of clothing from small grayscale images in the Fashion-MNIST dataset.

## Try it in Google Colab (no setup needed)

[fashion_mnist_cnn.ipynb - Colab](https://colab.research.google.com/github/rawanwehbe047-ops/FASHION-MNIST-CNN-classification/blob/main/fashion_mnist_cnn.ipynb)

## What this project does

- Downloads the Fashion-MNIST dataset.
- Resizes images to 16 × 16 pixels and converts them to tensors.
- Builds a CNN with two convolution layers and max-pooling.
- Trains the model for five epochs.
- Tracks loss and accuracy across epochs.
- Displays sample clothing images and model predictions.

## Dataset

Fashion-MNIST contains 70,000 grayscale images across 10 categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot. The standard split contains 60,000 training images and 10,000 test images.

Dataset source: [Fashion-MNIST on GitHub](https://github.com/zalandoresearch/fashion-mnist).

## Project files

- `fashion_mnist_cnn.ipynb` — notebook version, best for exploring the project in Colab.
- `fashion_mnist_cnn.py` — Python script version.
- `requirements.txt` — packages needed to run the script.

## Run locally

Use Python 3.10 or newer.

```bash
pip install -r requirements.txt
python fashion_mnist_cnn.py
```

The dataset downloads automatically the first time the script runs.

## Model notes

The model uses two convolution layers, ReLU activation, max-pooling, and a final linear layer that returns scores for 10 classes. It is trained using cross-entropy loss and stochastic gradient descent (SGD).

The development notebook showed approximately **88% accuracy after five epochs**. Results can vary with runtime and training settings. The notebook checks accuracy on the test split after each epoch, so treat this as a progress metric rather than a final independent evaluation.

## Tools

- Python
- PyTorch and Torchvision
- Matplotlib
- Google Colab / Jupyter Notebook

---
Created as a learning project to practice CNN image classification.
