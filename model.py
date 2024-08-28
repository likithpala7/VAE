import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.sample(mean, logvar)
        x_hat = self.decoder(z)
        return x_hat, mean, logvar
    
    def sample(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps*std

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_dim, 64, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*8*8, 128)
        self.fc2 = nn.Linear(128, 16)
        self.mean = nn.Linear(16, latent_dim)
        self.logvar = nn.Linear(16, latent_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        logvar = self.logvar(x)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(latent_dim, 16)
        self.fc2 = nn.Linear(16, 128)
        self.fc3 = nn.Linear(128, 128*8*8)
        self.reshape = nn.Unflatten(1, (128, 8, 8))
        self.conv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(64, output_dim, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.reshape(x)
        x = torch.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x
        