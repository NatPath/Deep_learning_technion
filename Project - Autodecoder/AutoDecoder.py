import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader
from utils import create_dataloaders

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super(Decoder, self).__init__()
        self.output_shape = output_shape

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_shape[0] * output_shape[1] * output_shape[2]),
            nn.Sigmoid()  # Output pixels are between 0 and 1
        )

    def forward(self, z):
        x_hat = self.decoder(z)
        x_hat = x_hat.view(-1, *self.output_shape)
        return x_hat

class AutoDecoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super(AutoDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.decoder = Decoder(latent_dim, output_shape)

    def forward(self, z):
        return self.decoder(z)

def train_auto_decoder(model, train_dl, optimizer, latents, device, epochs=10):
    model.train()
    reconstruction_loss = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for i, (indices, x) in enumerate(train_dl):
            x = x.to(device)
            z = latents[indices].to(device)  # Get the latent vectors for the current batch

            optimizer.zero_grad()
            x_hat = model(z)
            loss = reconstruction_loss(x_hat, x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")