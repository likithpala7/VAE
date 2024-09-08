import torch
from PIL import Image
from model import VAE
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import os
import argparse

# Load the arguments
parser = argparse.ArgumentParser()

parser.add_argument("--num_images", type=int, default=20)
parser.add_argument("--latent_dim", type=int, default=512)
parser.add_argument("--model_path", type=str, default="saved_models/vae_24.pth")

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
vae = VAE(3, parser.latent_dim).to(device)
vae.load_state_dict(torch.load(args.model_path))
vae.eval()

# Load 20 random images
paths = os.listdir("D:/downloads/img_align_celeba/img_align_celeba")
indices = np.random.randint(0, len(paths), parser.num_images)
images = []
for i in indices:
    image = Image.open(f"D:/downloads/img_align_celeba/img_align_celeba/{paths[i]}")
    image = image.resize((128, 128))
    image = transforms.ToTensor()(image)
    images.append(image)

# Reconstruct selected images
images = torch.stack(images).to(device)
with torch.no_grad():
    x_hat, mean, logvar = vae(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, x_hat)):
        image = image.permute(1, 2, 0) * 255
        image = image.cpu().numpy().astype("uint8")
        reconstructed_image = reconstructed_image.permute(1, 2, 0) * 255
        reconstructed_image = reconstructed_image.cpu().numpy().astype("uint8")

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(image)
        axs[0].set_title("Original Image")
        axs[1].imshow(reconstructed_image)
        axs[1].set_title("Reconstructed Image")

        axs[0].axis("off")
        axs[1].axis("off")

        plt.savefig(f"output/reconstructed_image_{i}.png")
        print(f"Reconstructed image saved to output/reconstructed_image_{i}.png")


# Generate random images
with torch.no_grad():
    z = torch.randn(args.num_images, args.latent_dim).to(device)
    images = vae.decoder(z).cpu()
    for i, image in enumerate(images):
        image = image.permute(1, 2, 0)*255
        image = image.numpy().astype("uint8")
        plt.clf()
        plt.imshow(image)
        plt.savefig(f"output/image_{i}.png")
        print(f"Image saved to output/image_{i}.png")