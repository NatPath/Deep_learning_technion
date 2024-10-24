import torch
import torch.nn as nn
from evaluate import reconstruction_loss as rec_loss
from AutoDecoder import DecoderCNN, visualize_reconstructions, visualize_from_latents
import torch.distributions as dist
import matplotlib.pyplot as plt

class VariationalAutoDecoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super(VariationalAutoDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        
        # Decoder network
        self.fc1 = nn.Linear(latent_dim, 512 * 4 * 4)
        self.decoder = DecoderCNN(512, output_shape[0])

    def forward(self, z):
        
        h = self.fc1(z)
        # print(h)
        h = h.view(-1, 512, 4, 4)  # Reshape to match the input of the CNN decoder
        # print(h)
        decoder_res = self.decoder(h)
        # print(decoder_res)
        res = decoder_res.view(-1, 28, 28)
        # print(res)
        return res * 255.0

def reparameterize(distribution, *args):
    if distribution == 'gaussian':
        mu, log_var = args
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    elif distribution == 'poisson':
        rate = args[0]
        return dist.Poisson(rate).rsample()
    elif distribution == 'uniform':
        low, high = args
        return dist.Uniform(low, high).rsample()
    else:
        raise ValueError("Unsupported distribution")

def kl_divergence(distribution, *args):
    if distribution == 'gaussian':
        mu, log_var = args
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    elif distribution == 'poisson':
        rate = args[0]
        return torch.sum(rate * (torch.log(rate) - 1) + torch.lgamma(rate + 1), dim=1)
    elif distribution == 'uniform':
        low, high = args
        return torch.sum(torch.log(high - low), dim=1)
    else:
        raise ValueError("Unsupported distribution")

def train_variational_auto_decoder(model, train_dl, optimizer, dist_params, device, distribution='gaussian', epochs=10, beta=1.0):
    model.train()
    # criterion = rec_loss
    criterion = nn.MSELoss()
    train_losses = []

    for epoch in range(epochs):
        total_train_loss = 0
        for i, (indices, x) in enumerate(train_dl):
            x = x.to(device).float()
            batch_size = x.size(0)
            
            # Get distribution parameters for this batch
            batch_dist_params = [param[indices].to(device) for param in dist_params]
            
            optimizer.zero_grad()
            z = reparameterize(distribution, *batch_dist_params)
            # print(z)
            # print(type(z))
            x_hat = model(z)
            # print(x_hat)
            reconstruction_loss = criterion(x_hat, x)
            # print(reconstruction_loss)
            kl_div = kl_divergence(distribution, *batch_dist_params).mean()
            # print(kl_div)
            # Total loss is reconstruction loss + beta * KL divergence
            loss = reconstruction_loss + beta * kl_div
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dl)
        train_losses.append(avg_train_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

    return train_losses

def visualize_vad_reconstructions(model, test_dl, dist_params, device, distribution='gaussian', num_samples=10):
    model.eval()
    with torch.no_grad():
        # Get a batch of test data
        indices, x = next(iter(test_dl))
        x = x.to(device).float()
        
        # Get distribution parameters for this batch
        batch_dist_params = [param[indices].to(device) for param in dist_params]
        
        # Generate latent vectors using reparameterization
        z = reparameterize(distribution, *batch_dist_params)
        
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

def visualize_vad_from_dist_params(model, dist_params, device, distribution='gaussian', num_samples=10):
    model.eval()
    with torch.no_grad():
        # Ensure we only take the number of samples requested
        batch_dist_params = [param[:num_samples].to(device) for param in dist_params]
        
        # Generate latent vectors using reparameterization
        z = reparameterize(distribution, *batch_dist_params)
        
        # Generate reconstructions from the sampled latent vectors
        x_hat = model(z)

        # Move tensors to CPU for visualization
        x_hat = x_hat.cpu()

        # Plot reconstructed images
        fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
        for i in range(num_samples):
            axes[i].imshow(x_hat[i].squeeze(), cmap='gray')
            axes[i].axis('off')
            axes[i].set_title('Generated')

        plt.tight_layout()
        plt.show()

def plot_learning_curve(train_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve (Variational AutoDecoder)')
    plt.legend()
    plt.grid(True)
    plt.show()
