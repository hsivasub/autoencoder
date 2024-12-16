import torch
import torch.nn as nn
import torch.optim as optim

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AnomalyDetector(nn.Module):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(140, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 140),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Instantiate the model
autoencoder = AnomalyDetector().to(device)

# Define the optimizer and loss function
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
loss_fn = nn.L1Loss()  # Equivalent to Mean Absolute Error (MAE)

# Summary function for PyTorch
def summary(model, input_size):
    from torchsummary import summary
    summary(model, input_size)

# Display the model summary
summary(autoencoder, (140,))
