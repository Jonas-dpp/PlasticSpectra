import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
import numpy as np
from winsound import Beep

# Define the dataset class for handling JSON data
class MaterialDataset(Dataset):
    def __init__(self, file_name):
        """
        Initialize the dataset by loading the JSON file and preprocessing the data.
        """
        with open(file_name, "r") as file:
            data = json.load(file)

        self.samples = []
        self.labels = []
        self.label_mapping = {}

        # Parse the JSON data and create samples and labels
        for idx, entry in enumerate(data):
            for material, pixels in entry.items():
                if material not in self.label_mapping:
                    self.label_mapping[material] = len(self.label_mapping)  # Assign unique label to each material
                label = self.label_mapping[material]

                for pixel_index, features in pixels.items():
                    # Combine RGB and HSV values as input features
                    rgb_values = list(map(int, features["RGB"].values()))
                    hsv_values = list(map(int, features["HSV"].values()))
                    input_features = rgb_values + hsv_values
                    self.samples.append(input_features)
                    self.labels.append(label)

        # Convert to PyTorch tensors
        self.samples = torch.tensor(self.samples, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        Retrieve a single sample and its label by index.
        """
        return self.samples[index], self.labels[index]

# Define the neural network
class MaterialClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        """
        Initialize the neural network layers.
        """
        super(MaterialClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Add dropout to reduce overfitting

    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after the first layer
        x = self.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout after the second layer
        x = self.fc3(x)
        return x

# Load the dataset
file_name = "Dataset.json"
dataset = MaterialDataset(file_name)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize the neural network, loss function, and optimizer
input_size = 6  # 3 RGB + 3 HSV values
num_classes = len(dataset.label_mapping)
model = MaterialClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Add weight decay to prevent overfitting

# Initialize TensorBoard writer
writer = SummaryWriter("runs/MaterialClassification")

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    epoch_loss = 0.0
    model.train()  # Ensure the model is in training mode

    for inputs, labels in data_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        epoch_loss += loss.item()

    # Calculate average loss for the epoch
    avg_loss = epoch_loss / len(data_loader)

    # Log the epoch loss to TensorBoard
    writer.add_scalar("Loss/train", avg_loss, epoch)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Early stopping condition (if the loss stagnates)
    if avg_loss < 1e-4:  # Stop if the average loss is very low
        print("Early stopping as loss is below threshold.")
        break

# Save the trained model
torch.save(model.state_dict(), "material_classifier.pth")
print("Model training complete and saved.")

# Save the label mapping
with open("label_mapping.json", "w") as f:
    json.dump(dataset.label_mapping, f)
print("Label mapping saved.")
Beep(650, 500) # Notification
