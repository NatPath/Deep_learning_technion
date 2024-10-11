import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader
from utils import create_dataloaders

# Define activations
ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax,
    "lrelu": nn.LeakyReLU,
    "none": nn.Identity,
    None: nn.Identity,
}

# Define DecoderCNN class
class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # Implement transposed convolutions for Decoder
        conv_params = dict(kernel_size=4, stride=2, padding=1, bias=False)
        activation_type = 'relu'
        activation_params = dict(inplace=True)

        channels_list = [512, 256, 128]
        modules.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=channels_list[0], **conv_params))
        modules.append(nn.BatchNorm2d(channels_list[0]))
        modules.append(ACTIVATIONS[activation_type](**activation_params))

        prev_channels = channels_list[0]
        for channels in channels_list[1:]:
            modules.append(nn.ConvTranspose2d(in_channels=prev_channels, out_channels=channels, **conv_params))
            modules.append(nn.BatchNorm2d(channels))
            modules.append(ACTIVATIONS[activation_type](**activation_params))
            prev_channels = channels
        modules.append(nn.ConvTranspose2d(prev_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        #modules.append(nn.ConvTranspose2d(prev_channels, out_channels, **conv_params))
        #modules.append(nn.Sigmoid())  # Output pixel values between 0 and 1

        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        return self.cnn(h)

# Define AutoDecoder class using DecoderCNN
class AutoDecoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super(AutoDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)  # Project latent vector to feature map size
        self.decoder = DecoderCNN(512, output_shape[0])

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 512, 4, 4)  # Reshape to match the input of the CNN decoder
        return self.decoder(h)

def train_auto_decoder(model, train_dl, optimizer, latents, device, epochs=10):
    model.train()
    reconstruction_loss = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for i, (indices, x) in enumerate(train_dl):
            x = x.to(device).float()
            z = latents[indices].to(device)  # Get the latent vectors for the current batch
            optimizer.zero_grad()
            x_hat = model(z)
            loss = reconstruction_loss(x_hat, x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

'''

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
            #nn.Sigmoid()*255  # Output pixels are between 0 and 1
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

'''