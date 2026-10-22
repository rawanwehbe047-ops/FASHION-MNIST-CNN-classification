"""Fashion-MNIST image classification with a simple PyTorch CNN.

Run:
    python fashion_mnist_cnn.py

The first run downloads Fashion-MNIST into ./data.
"""
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

IMAGE_SIZE = 16
BATCH_SIZE = 100
EPOCHS = 5
LEARNING_RATE = 0.1
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


class CNN(nn.Module):
    """Two convolution layers followed by a 10-class output layer."""
    def __init__(self, out_1=16, out_2=32, number_of_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, out_1, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(out_1, out_2, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(out_2 * 4 * 4, number_of_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def main():
    torch.manual_seed(0)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    train_data = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training images: {len(train_data):,}")
    print(f"Test images: {len(test_data):,}")
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    costs, accuracies = [], []
    start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        costs.append(total_loss)

        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
        accuracy = correct / len(test_data)
        accuracies.append(accuracy)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss sum: {total_loss:.2f} | Test accuracy: {accuracy:.2%}")

    print(f"Training finished in {time.time() - start:.1f} seconds.")
    print("Note: test accuracy is checked after every epoch for tracking.")

    fig, ax1 = plt.subplots()
    epochs = range(1, EPOCHS + 1)
    ax1.plot(epochs, costs, marker="o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Sum of batch losses")
    ax2 = ax1.twinx()
    ax2.plot(epochs, accuracies, marker="s")
    ax2.set_ylabel("Test accuracy")
    plt.title("Fashion-MNIST CNN training")
    fig.tight_layout()
    plt.show()

    model.eval()
    images, labels = next(iter(test_loader))
    with torch.no_grad():
        predictions = model(images).argmax(dim=1)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap="gray")
        ax.set_title(f"Pred: {CLASS_NAMES[predictions[i]]}\nTrue: {CLASS_NAMES[labels[i]]}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
