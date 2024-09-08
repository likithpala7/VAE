import torch.nn as nn
from torch.utils.data import Dataset


"""
Dataset class for the VAE
- ds: the dataset object
- transform: the transformation to apply to the images
"""
class VAEDataset(Dataset):
    def __init__(self, ds, transform=None):
        self.ds = ds['train']
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example = self.ds[idx]
        image = example['image']
        if image.mode == "L":
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image