import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, TensorDataset
#import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from autoencoder.AutoencoderPythonFiles.model_AE import *
from autoencoder.AutoencoderPythonFiles.model_AE_SingleDiffusion import *
from metrics import *
from plots import *

import random

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# Download and load the dataset
dataframe = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
raw_data = dataframe.values
dataframe.head()

# The last element contains the labels
labels = raw_data[:, -1]

# The other data points are the electrocardiogram data
data = raw_data[:, 0:-1]

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=21
)

# Normalize the data using min-max scaling
min_val = torch.min(torch.tensor(train_data))
max_val = torch.max(torch.tensor(train_data))

train_data = (torch.tensor(train_data) - min_val) / (max_val - min_val)
test_data = (torch.tensor(test_data) - min_val) / (max_val - min_val)

# Convert the labels to boolean tensors
train_labels = torch.tensor(train_labels, dtype=torch.bool)
test_labels = torch.tensor(test_labels, dtype=torch.bool)

# Separate normal and anomalous data
normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]


# Convert data to PyTorch tensors and move to the device
normal_train_data = torch.tensor(normal_train_data, dtype=torch.float32).to(device)
normal_test_data = torch.tensor(normal_test_data, dtype=torch.float32).to(device)
anomalous_test_data = torch.tensor(anomalous_test_data, dtype=torch.float32).to(device)

# Create DataLoaders
batch_size = 512
train_dataset = TensorDataset(normal_train_data, normal_train_data)  # Inputs == Targets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(normal_test_data, normal_test_data)  # Inputs == Targets for testing
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Training loop
epochs = 20
for epoch in range(epochs):
    autoencoder.train()  # Training mode
    train_loss = 0.0

    for inputs, targets in train_loader:
        # Forward pass
        outputs = autoencoder(inputs)
        loss = loss_fn(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate training loss
        train_loss += loss.item() * inputs.size(0)

    # Average training loss
    train_loss /= len(train_loader.dataset)

    # Validation loop
    autoencoder.eval()  # Evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = autoencoder(inputs)
            loss = loss_fn(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    # Average validation loss
    val_loss /= len(test_loader.dataset)

    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")


plot_reconstruction_error(autoencoder_model=autoencoder,data=normal_test_data,device=device,type='train')
plot_reconstruction_error(autoencoder_model=autoencoder,data=anomalous_test_data,device=device, type='test')

loss_distribution(autoencoder_model=autoencoder,data=normal_train_data,device=device,type='train')

# Define the threshold as mean + std of train loss
threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)

loss_distribution(autoencoder_model=autoencoder,data=anomalous_test_data,device=device,type='test')

# Make predictions and evaluate
with torch.no_grad():
    preds = predict(autoencoder, test_data, threshold,device)

print_stats(preds, test_labels)
