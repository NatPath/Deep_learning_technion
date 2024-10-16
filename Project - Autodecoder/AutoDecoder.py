import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import create_dataloaders
from evaluate import reconstruction_loss
import matplotlib.pyplot as plt

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

        # Fix 32 to 28 output size
        modules.append(nn.ConvTranspose2d(out_channels, out_channels, kernel_size=1, stride=1, padding=2, bias=False))
        #modules.append(nn.ConvTranspose2d(prev_channels, out_channels, **conv_params))
        modules.append(nn.Sigmoid())  # Output pixel values between 0 and 1

        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        return self.cnn(h)

# Define AutoDecoder class using DecoderCNN
class AutoDecoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super(AutoDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.fc1 = nn.Linear(latent_dim, 512 * 4 * 4)  # Project latent vector to feature map size
        self.decoder = DecoderCNN(512, output_shape[0])

    def forward(self, z):
        h = self.fc1(z)
        h = h.view(-1, 512, 4, 4)  # Reshape to match the input of the CNN decoder
        decoder_res=self.decoder(h)
        res= decoder_res.view(-1,28,28)
        return res*255.0

def train_auto_decoder(model, train_dl, optimizer, train_latents, device, epochs=10):
    model.train()
    criterion = reconstruction_loss
    train_losses = []

    for epoch in range(epochs):
        total_train_loss = 0
        for i, (indices, x) in enumerate(train_dl):
            x = x.to(device).float() 
            z = train_latents[indices].to(device)
            
            optimizer.zero_grad()
            x_hat = model(z)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dl)
        train_losses.append(avg_train_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

    return train_losses

def plot_learning_curve(train_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_reconstructions(model, test_dl, latents, device, num_samples=10):
    model.eval()
    with torch.no_grad():
        # Get a batch of test data
        indices, x = next(iter(test_dl))
        x = x.to(device).float()
        z = latents[indices].to(device)

        # Generate reconstructions
        x_hat = model(z)

        # Move tensors to CPU for visualization
        x = x.cpu()
        x_hat = x_hat.cpu()

        # Plot original and reconstructed images
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 4))
        for i in range(num_samples):
            axes[0, i].imshow(x[i].squeeze(), cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title('Original')

            axes[1, i].imshow(x_hat[i].squeeze(), cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title('Reconstructed')

        plt.tight_layout()
        plt.show()

# class Decoder(nn.Module):
#     def __init__(self, latent_dim, output_shape):
#         super(Decoder, self).__init__()
#         self.output_shape = output_shape

#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, output_shape[0] * output_shape[1] * output_shape[2]),
#             #nn.Sigmoid()*255  # Output pixels are between 0 and 1
#         )

#     def forward(self, z):
#         x_hat = self.decoder(z)
#         x_hat = x_hat.view(-1, *self.output_shape)
#         return x_hat

# class AutoDecoder(nn.Module):
#     def __init__(self, latent_dim, output_shape):
#         super(AutoDecoder, self).__init__()
#         self.latent_dim = latent_dim
#         self.decoder = Decoder(latent_dim, output_shape)

#     def forward(self, z):
#         return self.decoder(z)
