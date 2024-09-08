import torch
import torch.nn as nn
from torchvision import models


"""
Define the VAE model with encoder and decoder components
- input_dim: the number of input channels
- latent_dim: the dimension of the latent space
"""
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


"""
Define the encoder for the VAE with 5 down-convs
- input_dim: the number of input channels
- latent_dim: the dimension of the latent space
"""
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_dim, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.flatten = nn.Flatten()
        self.mean = nn.Linear(512*4*4, latent_dim)
        self.logvar = nn.Linear(512*4*4, latent_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))
        x = self.flatten(x)
        mean = self.mean(x)
        logvar = self.logvar(x)
        return mean, logvar


"""
Define the decoder for the VAE with 5 up-convs
- latent_dim: the dimension of the latent space
- output_dim: the number of output channels
"""
class Decoder(nn.Module):

    def __init__(self, latent_dim, output_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(latent_dim, 512*4*4)
        self.reshape = nn.Unflatten(1, (512, 4, 4))
        self.conv2 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.final = nn.ConvTranspose2d(32, output_dim, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.reshape(self.fc1(x))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.sigmoid(self.final(x))

        return x