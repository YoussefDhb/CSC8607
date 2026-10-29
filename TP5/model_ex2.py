import torch.nn as nn
from torchvision import models

# 1. Chargement de ResNet50 pré-entraîné
resnet_pretrained = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# 2. Suppression des deux dernières couches (AdaptiveAvgPool2d et Linear)
# On veut conserver les dimensions spatiales pour l'attention (pas un vecteur 1D global)
cnn_features = nn.Sequential(*list(resnet_pretrained.children())[:-2])

# 3. Gel des poids (Transfer Learning)
for param in cnn_features.parameters():
    param.requires_grad = False


def check_cnn():
    """afficher les dernières couches retirées, le nombre de paramètres gelés et la forme de sortie de cnn_features."""
    import torch
    print("Couches retirées :", [type(m).__name__ for m in list(resnet_pretrained.children())[-2:]])
    n_total = sum(p.numel() for p in cnn_features.parameters())
    n_trainable = sum(p.numel() for p in cnn_features.parameters() if p.requires_grad)
    print(f"Paramètres de cnn_features : {n_total:,} | entraînables : {n_trainable:,}")
    x = torch.randn(32, 3, 224, 224)
    with torch.no_grad():
        out = cnn_features(x)
    print(f"Entrée : {tuple(x.shape)} -> Sortie : {tuple(out.shape)}")


if __name__ == "__main__":
    check_cnn()