import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


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
