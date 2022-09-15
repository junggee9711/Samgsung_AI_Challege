from turtle import width
import torch
import torch.nn as nn
import torch.nn.functional as F

# VAE model
class VAE(nn.Module):
    def __init__(self, height, width, hidden_size_1, hidden_size_2, latent_size):
        super(VAE, self).__init__()
        self.img_size = height*width
        self.height = height
        self.width = width

        self.fc1 = nn.Linear(self.img_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc31 = nn.Linear(hidden_size_2, latent_size)
        self.fc32 = nn.Linear(hidden_size_2, latent_size)

        self.fc4 = nn.Linear(latent_size, hidden_size_2)
        self.fc5 = nn.Linear(hidden_size_2, hidden_size_1)
        self.fc6 = nn.Linear(hidden_size_1, self.img_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h3 = F.relu(self.fc4(z))
        h4 = F.relu(self.fc5(h3))
        return torch.sigmoid(self.fc6(h4))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.img_size))
        z = self.reparameterize(mu, logvar)
        z = self.decode(z)
        return z.view(-1, self.height, self.width), mu, logvar