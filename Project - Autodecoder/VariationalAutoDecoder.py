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
    elif distribution == 'exponential':
        log_rate = args[0]
        rate = torch.exp(log_rate)
        eps = torch.rand_like(log_rate) + 1e-6
        return -torch.log(eps) / rate
    elif distribution == 'uniform':
        low, log_length = args
        length = torch.exp(log_length)
        eps = torch.rand_like(low)
        return low + (length) * eps
    elif distribution == 'multivariate_gaussian':
        mu, L = args
        eps = torch.randn_like(mu)
        return mu + torch.matmul(L, eps.unsqueeze(-1)).squeeze(-1)
    elif distribution == 'laplace':
        mu, log_b = args
        b = torch.exp(log_b)
        u = torch.rand_like(mu) - 0.5
        epsilon = 1e-6
        return mu - b * torch.sign(u) * torch.log(1 - 2 * torch.abs(u)+epsilon)
    else:
        raise ValueError("Unsupported distribution")

def kl_divergence(distribution, *args):
    if distribution == 'gaussian':
        mu, log_var = args
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    elif distribution == 'exponential':
        log_rate = args[0]
        return torch.sum(log_rate + torch.exp(-log_rate) - 1, dim=1)
    elif distribution == 'uniform':
        low, log_length = args
        length = torch.exp(log_length)
        return torch.sum(torch.log(length), dim=1)
    elif distribution == 'multivariate_gaussian':
        mu, L = args
        n = mu.size(-1)
        log_det_cov = 2 * torch.sum(torch.log(torch.abs(torch.diagonal(L, dim1=-2, dim2=-1) + 1e-6)), dim=-1)
        trace_cov = torch.sum(torch.square(L), dim=(-2, -1))
        mahalanobis = torch.sum(torch.square(mu), dim=-1)
        kl_div = 0.5 * (log_det_cov - n + trace_cov + mahalanobis)
        return kl_div
    elif distribution == 'laplace':
        log_abs_mu, log_b = args
        abs_mu = torch.exp(log_abs_mu)
        b = torch.exp(log_b)
        return torch.sum(b*torch.exp(-abs_mu/b)-1-log_b+abs_mu, dim=1)
        # return torch.sum(b*torch.exp(-torch.abs(mu)/b)-log_b-1+torch.abs(mu), dim=1)
    else:
        raise ValueError("Unsupported distribution")

def train_variational_auto_decoder(model, train_dl, optimizer, dist_params, device, distribution='gaussian', epochs=10, beta=1.0):
    model.train()
    criterion = rec_loss
    train_losses = []

    for epoch in range(epochs):
        total_train_loss = 0
        for i, (indices, x) in enumerate(train_dl):
            x = x.to(device).float()
            batch_size = x.size(0)
            
            if distribution in ['multivariate_gaussian', 'laplace']:
                batch_dist_params = [param[indices].to(device) for param in dist_params]
            else:
                batch_dist_params = [param[indices].to(device) for param in dist_params]
            
            optimizer.zero_grad()
            z = reparameterize(distribution, *batch_dist_params)
            x_hat = model(z)
            reconstruction_loss = criterion(x_hat, x)
            kl_div = kl_divergence(distribution, *batch_dist_params).mean()
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

def evaluate_vad_model(model, test_dl, opt, dist_params, epochs, device, distribution='gaussian'):
    reconstruction_loss = rec_loss
    
    for epoch in range(epochs):
        for i, x in test_dl:
            i = i.to(device)
            x = x.to(device)
            if distribution == 'gaussian':
                mu, log_var = dist_params
                latents = reparameterize(distribution, mu[i], log_var[i])
            elif distribution == 'exponential':
                rate = dist_params[0]
                latents = reparameterize(distribution, rate[i])
            elif distribution == 'uniform':
                low, log_length = dist_params
                latents = reparameterize(distribution, low[i], log_length[i])
            elif distribution == 'multivariate_gaussian':
                mu, L = dist_params
                latents = reparameterize(distribution, mu[i], L[i])
            elif distribution == 'laplace':
                mu, log_b = dist_params
                latents = reparameterize(distribution, mu[i], log_b[i])
            x_rec = model(latents)
            loss = reconstruction_loss(x, x_rec)
            if torch.isnan(loss):
                print(f"Loss is nan at epoch {epoch}, batch {i}")
                break
            opt.zero_grad()
            loss.backward()
            opt.step()

    losses = []
    with torch.no_grad():
        for i, x in test_dl:
            i = i.to(device)
            x = x.to(device)
            if distribution == 'gaussian':
                mu, log_var = dist_params
                latents = reparameterize(distribution, mu[i], log_var[i])
            elif distribution == 'exponential':
                rate = dist_params[0]
                latents = reparameterize(distribution, rate[i])
            elif distribution == 'uniform':
                low, log_length = dist_params
                latents = reparameterize(distribution, low[i], log_length[i])
            elif distribution == 'multivariate_gaussian':
                mu, L = dist_params
                latents = reparameterize(distribution, mu[i], L[i])
            elif distribution == 'laplace':
                mu, log_b = dist_params
                latents = reparameterize(distribution, mu[i], log_b[i])
            x_rec = model(latents)
            loss = reconstruction_loss(x, x_rec)
            losses.append(loss.item())

        final_loss = sum(losses) / len(losses)

    return final_loss

def generate_latents_from_dist_params(dist_params, distribution='gaussian'):
    if distribution == 'gaussian':
        mu, log_var = dist_params
        return reparameterize(distribution, mu, log_var)
    elif distribution == 'exponential':
        rate = dist_params[0]
        return reparameterize(distribution, rate)
    elif distribution == 'uniform':
        low, log_length = dist_params
        return reparameterize(distribution, low, log_length)
    elif distribution == 'multivariate_gaussian':
        mu, L = dist_params
        return reparameterize(distribution, mu, L)
    elif distribution == 'laplace':
        mu, log_b = dist_params
        return reparameterize(distribution, mu, log_b)
    else:
        raise ValueError("Unsupported distribution")

class LatentGenerator(nn.Module):
    def __init__(self, dist_params, distribution='gaussian'):
        super(LatentGenerator, self).__init__()
        self.dist_params = nn.ParameterList([nn.Parameter(param) for param in dist_params])
        self.distribution = distribution

    def forward(self):
        return generate_latents_from_dist_params(self.dist_params, self.distribution)
