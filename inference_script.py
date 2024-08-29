import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from model import VAE
import matplotlib.pyplot as plt

num_images = 20
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
model_path = Path("saved_models/vae_3.pth")
vae = VAE(3, 2).to(device)
vae.load_state_dict(torch.load(model_path))
vae.eval()

# Generate images
with torch.no_grad():
    z = torch.randn(num_images, 2).to(device)
    images = vae.decoder(z).cpu()
    for i, image in enumerate(images):
        image = (image + 1.0) * 127.5
        image = image.permute(1, 2, 0).numpy().astype("uint8")
        image = Image.fromarray(image)
        # plt.imshow(image)
        # plt.show()
        image.save(f"output/image_{i}.png")
        print(f"Image saved to output/image_{i}.png")