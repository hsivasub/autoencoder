import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Prediction function
def predict(model, data, threshold,device):
    model.eval()
    with torch.no_grad():
        reconstructions = model(data.to(device).float())
        loss = torch.mean(torch.abs(reconstructions - data.to(device)), dim=1)
        return loss < threshold

# Print stats function
def print_stats(predictions, labels):
    print("Accuracy = {:.4f}".format(accuracy_score(labels.cpu(), predictions.cpu())))
    print("Precision = {:.4f}".format(precision_score(labels.cpu(), predictions.cpu())))
    print("Recall = {:.4f}".format(recall_score(labels.cpu(), predictions.cpu())))