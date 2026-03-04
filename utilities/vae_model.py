import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAutoencoder(nn.Module):
    def __init__(self, embedding_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc31 = nn.Linear(32, latent_dim)
        self.fc32 = nn.Linear(32, latent_dim)
        self.fc4 = nn.Linear(latent_dim, 32)
        self.fc5 = nn.Linear(32, 64)
        self.fc6 = nn.Linear(64, embedding_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc4(z))
        h4 = F.relu(self.fc5(h3))
        return self.fc6(h4)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.embedding_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    
def vae_loss_function(recon_x, x, mu, logvar):
    squared_error = (recon_x - x) ** 2
    mask = x == 0
    squared_error[mask] = 0
    recon_loss = torch.mean(squared_error)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

def calculate_reconstruction_error(samples, model, loss_function):
    model.eval()
    reconstruction_errors = []
    with torch.no_grad():
        for sample in samples:
            test_sample = torch.Tensor(sample).view(1, -1)
            recon_sample, mu, logvar = model(test_sample)
            reconstruction_error = loss_function(recon_sample, test_sample, mu, logvar)
            reconstruction_errors.append(reconstruction_error.item())
    return reconstruction_errors
