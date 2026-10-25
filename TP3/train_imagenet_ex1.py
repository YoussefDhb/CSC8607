import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

data_dir = "/mnt/ssd/imagenet/2012"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# 1. Pipeline d'entraînement (avec Data Augmentation)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(227),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# 2. Pipeline de validation (déterministe)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(227),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def get_dataloaders(batch_size=256, n_workers=8, subset_size=None):
    print("Chargement d'ImageNet...")
    train_dataset = ImageNet(root=data_dir, split="train", transform=train_transform)
    val_dataset   = ImageNet(root=data_dir, split="val",   transform=val_transform)

    # Optionnel : sous-ensemble pour tester rapidement le code en séance
    if subset_size is not None:
         generator = torch.Generator().manual_seed(42)
         train_indices = torch.randperm(len(train_dataset), generator=generator)[:subset_size]
         val_indices   = torch.randperm(len(val_dataset), generator=generator)[:subset_size // 4]
         train_dataset = Subset(train_dataset, train_indices)
         val_dataset   = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True)
    return train_loader, val_loader


def check_data():
    """vérifier le chargement d'ImageNet et les transformations."""
    train_loader, val_loader = get_dataloaders(batch_size=256, subset_size=1000)
    print(f"Train : {len(train_loader.dataset)} images | Val : {len(val_loader.dataset)} images")
    images, labels = next(iter(train_loader))
    print(f"Batch train : images {tuple(images.shape)}, labels {tuple(labels.shape)}")
    print(f"Après Normalize : moyenne = {images.mean():.3f}, écart-type = {images.std():.3f}")
    images, _ = next(iter(val_loader))
    print(f"Batch val   : images {tuple(images.shape)}")


if __name__ == "__main__":
    check_data()