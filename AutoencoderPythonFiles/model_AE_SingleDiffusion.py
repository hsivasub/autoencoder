import torch
import torch.nn as nn
import torch.optim as optim

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Encoder, self).__init__()
        self.dense = nn.Linear(input_dim, encoding_dim)
        self.norm = nn.LayerNorm(encoding_dim)
        self.attention = nn.MultiheadAttention(encoding_dim, num_heads=1, batch_first=True)

    def forward(self, x):
        x = torch.relu(self.dense(x))
        x = self.norm(x)

        # Expand dimensions for attention layer
        x = x.unsqueeze(1)  # Shape: (batch_size, seq_len=1, encoding_dim)
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output

        return x.squeeze(1)  # Remove the seq_len dimension

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, encoding_dim, output_dim):
        super(Decoder, self).__init__()
        self.dense = nn.Linear(encoding_dim, output_dim)
        self.norm = nn.LayerNorm(encoding_dim)
        self.attention = nn.MultiheadAttention(encoding_dim, num_heads=1, batch_first=True)

    def forward(self, x):
        x = self.norm(x)
        x = x.unsqueeze(1)

        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output

        x = x.squeeze(1)
        x = torch.sigmoid(self.dense(x))
        return x

# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, encoding_dim)
        self.decoder = Decoder(encoding_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

# Model parameters
input_dim = 140  # Adjust this based on your data
encoding_dim = 8


# Instantiate the autoencoder
autoencoder_SingleDiffusion = Autoencoder(input_dim, encoding_dim).to(device)
loss_fn = nn.L1Loss()
optimizer = optim.Adam(autoencoder_SingleDiffusion.parameters(), lr=0.001)