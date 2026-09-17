import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

# 1. Load the LFW Dataset (reused from the Eigenfaces part)
print("Fetching LFW dataset...")
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

X = lfw_people.images
Y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print(f"X_min: {X.min():.2f}, X_max: {X.max():.2f}")  # Usually 0.0 to 1.0 or ~255 depending on sklearn version

# 2. Split and Preprocess for PyTorch
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# PyTorch expects dimensions to be [Batch_size, Channels, Height, Width]
# Since these are grayscale images, we add a channel dimension of 1
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Convert NumPy arrays to PyTorch Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 3. Define the CNN Architecture
class LFWClassifier(nn.Module):
    def __init__(self, num_classes):
        super(LFWClassifier, self).__init__()
        # First Conv layer: 1 input channel (grayscale), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # Second Conv layer: 32 input channels, 32 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)

        # Max pooling to reduce spatial dimensions and prevent overfitting/massive weight matrices
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # LazyLinear automatically infers the input size based on the output of the previous layer
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply Conv1 + ReLU + Pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply Conv2 + ReLU + Pooling
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the tensor for the dense layers
        x = torch.flatten(x, 1)

        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"\nTraining on device: {device}")

model = LFWClassifier(num_classes=n_classes).to(device)

# PyTorch's CrossEntropyLoss expects class indices and applies Softmax internally (equivalent to sparse categorical cross entropy)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Run a dummy batch through the model once to initialize the LazyLinear layer
_ = model(X_train_tensor[:1].to(device))

# 4. Training Loop
epochs = 20
print("\nStarting Training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(
        f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {100 * correct / total:.2f}%")

# 5. Evaluation Phase
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nTest Accuracy on {total} images: {100 * correct / total:.2f}%")