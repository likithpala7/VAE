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
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

num_epochs = 20
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128

print("Loading dataset...")
ds = load_dataset("UCSC-VLAA/Recap-COCO-30K")
print("Dataset loaded.")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0], [1])
])

dataset = VAEDataset(ds, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

vae = VAE(3, 64).cuda()
optimizer = Adam(vae.parameters(), lr=1e-4)
mse_loss = nn.MSELoss()
scaler = GradScaler()
# scheduler = CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5)

losses = []
for epoch in range(num_epochs):
    vae.train()
    epoch_loss = 0
    epoch_mse = 0
    epoch_kl = 0
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        optimizer.zero_grad()
        images = batch
        images = images.to(device)
        x_hat, mean, logvar = vae(images)
        mse = mse_loss(x_hat, images)
        kl = -0.5 * (1 + logvar - torch.square(mean) - torch.exp(logvar)).sum()
        loss = mse + kl
        epoch_loss += loss.item()
        epoch_mse += mse.item()
        epoch_kl += kl.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    epoch_loss /= len(dataloader)
    epoch_mse /= len(dataloader)
    epoch_kl /= len(dataloader)
    losses.append(epoch_loss)
    print(f"Epoch {epoch+1} loss: {epoch_loss}, mse: {epoch_mse}, kl: {epoch_kl}")
    torch.save(vae.state_dict(), f"saved_models/vae_{epoch+1}.pth")