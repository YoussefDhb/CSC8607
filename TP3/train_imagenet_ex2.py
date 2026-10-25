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

#Etape 1
class MLP(nn.Module):
    def __init__(self, input_size=3 * 227 * 227, hidden_size=256, num_classes=1000):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.classifier(x)

#Etape 2
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_data():
    """vérifier le chargement d'ImageNet et les transformations."""
    train_loader, val_loader = get_dataloaders(batch_size=256, subset_size=1000)
    print(f"Train : {len(train_loader.dataset)} images | Val : {len(val_loader.dataset)} images")
    images, labels = next(iter(train_loader))
    print(f"Batch train : images {tuple(images.shape)}, labels {tuple(labels.shape)}")
    print(f"Après Normalize : moyenne = {images.mean():.3f}, écart-type = {images.std():.3f}")
    images, _ = next(iter(val_loader))
    print(f"Batch val   : images {tuple(images.shape)}")


def show_architecture():
    """afficher le nombre de paramètres du MLP et d'AlexNet, et la taille de sortie de chaque couche d'AlexNet."""
    print(f"MLP : {count_parameters(MLP()):,} paramètres")
    print(f"AlexNet : {count_parameters(AlexNet()):,} paramètres")

    x = torch.zeros(1, 3, 227, 227)
    print("\nTailles de sortie d'AlexNet (une image) :")
    for layer in AlexNet().features:
        x = layer(x)
        if not isinstance(layer, nn.ReLU):
            print(f"{layer.__class__.__name__:10s} -> {tuple(x.shape[1:])} | {x.numel() * 4 / 1024:.2f} KB")


if __name__ == "__main__":
    show_architecture()