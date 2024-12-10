import torch
import matplotlib.pyplot as plt
import numpy as np

# Visualize normal test data reconstruction

def plot_reconstruction_error(autoencoder_model,data,device,type):
    with torch.no_grad():
        encoded_data = autoencoder_model.encoder(data.to(device))
        decoded_data = autoencoder_model.decoder(encoded_data).cpu().numpy()

    plt.plot(data[0].cpu().numpy(), 'b')
    plt.plot(decoded_data[0], 'r')
    plt.fill_between(np.arange(140), decoded_data[0], data[0].cpu().numpy(), color='lightcoral')
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    if type=='train':
        plt.title('Train Input Vs Reconstruction')
    else:
        plt.title('Test Input Vs Reconstruction')
    plt.show()


def loss_distribution(autoencoder_model,data,device,type):
    autoencoder_model.eval()
    with torch.no_grad():
        reconstructions = autoencoder_model(data.to(device))
        loss = torch.mean(torch.abs(reconstructions - data.to(device)), dim=1).cpu().numpy()
    plt.hist(loss, bins=50)
    
    plt.ylabel("Number of examples")
    if type=='train':
        plt.xlabel("Train Loss")
        plt.title('Train Loss Distribution')
    else:
        plt.xlabel("Test Loss")
        plt.title('Test Loss Distribution')
    plt.show()