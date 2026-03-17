import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Load dataset

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


# Simple neural network

model = nn.Sequential(
    nn.Flatten(),
     nn.Linear(28*28, 128),
     nn.ReLU(),
     nn.Linear(128, 10)
     )

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop

for epoch in range(3):
    for images, labels in train_loader:
        predictions = model(images)
        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} completed")


# Test on one digit

image, label = train_dataset[0]

with torch.no_grad():
    prediction = model(image.unsqueeze(0))
    predicted_digit = prediction.argmax().item()

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Predictions: {predicted_digit} | Actual: {label}")
plt.show()