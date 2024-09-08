from datasets import load_dataset
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from dataloader import VAEDataset
from tqdm import tqdm
from model import VAE
from torch.optim import Adam
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

"""
Defining hyperparameters
"""

parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--beta", type=int, default=4)
parser.add_argument("--latent_dim", type=int, default=512)
parser.add_argument("--learning_rate", type=float, default=1e-3)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"


# Loading the dataset
print("Loading dataset...")
ds = load_dataset("D:/downloads/img_align_celeba/img_align_celeba")
print("Dataset loaded.")

"""
Define transformations and dataloader
"""
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = VAEDataset(ds, transform)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

"""
Define model, optimizer, and loss function
"""

vae = VAE(3, args.latent_dim).cuda()
optimizer = Adam(vae.parameters(), lr=args.learning_rate)
mse_loss = nn.MSELoss(reduction="sum")
scaler = GradScaler()
beta = args.beta

"""
Training loop
"""

losses = []
for epoch in range(args.num_epochs):
    vae.train()
    epoch_loss = 0
    epoch_mse = 0
    epoch_kl = 0

    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        optimizer.zero_grad()
        images = batch
        images = images.to(device)

        with autocast(device_type=device):
            x_hat, mean, logvar = vae(images)
            mse = mse_loss(x_hat, images)
            kl = -0.5 * torch.sum(1 + logvar - torch.square(mean) - torch.exp(logvar))
            kl_scaled = beta * kl
            loss = mse + kl_scaled

        epoch_loss += loss.item()
        epoch_mse += mse.item()
        epoch_kl += kl_scaled.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    epoch_loss /= len(dataloader)
    epoch_mse /= len(dataloader)
    epoch_kl /= len(dataloader)

    losses.append(epoch_loss)
    print(f"Epoch {epoch+1} loss: {epoch_loss}, mse: {epoch_mse}, kl: {epoch_kl}")
    torch.save(vae.state_dict(), f"saved_models/vae_{epoch+1}.pth")

plt.plot(losses)
plt.show()