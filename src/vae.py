import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=4096, latent_dim=128):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc1_bn = nn.BatchNorm1d(hidden_dim)

        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_mu_bn = nn.BatchNorm1d(latent_dim)

        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logbar_bn = nn.BatchNorm1d(latent_dim)

        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc3_bn = nn.BatchNorm1d(hidden_dim)

        self.fc4 = nn.Linear(hidden_dim, input_dim)
        self.fc4_bn = nn.BatchNorm1d(input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1_bn(self.fc1(x)))
        mu = self.fc2_mu_bn(self.fc2_mu(h1))
        logvar = self.fc2_logbar_bn(self.fc2_logvar(h1))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3_bn(self.fc3(z)))
        return self.fc4_bn(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar, beta):
    # MSE Loss
    mse_loss = F.mse_loss(recon_x, x, reduction='sum')
    # KL Divergence
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse_loss + beta * kld_loss