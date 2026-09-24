import os
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import random

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
random.seed(0)

# Normalisation "classique" pour CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# CHARGEMENT DES DONNÉES (à compléter)
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Ne pas monopoliser les CPUs sur Slurm
def get_num_workers(default=2, cap=4):
    try:
        n = int(os.getenv("SLURM_CPUS_PER_TASK", default))
    except Exception:
        n = default
    return max(0, min(cap, n))

num_workers = get_num_workers()

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True
)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Image 32x32 avec 3 canaux (RGB) -> aplatie
        self.fc1 = nn.Linear(3072, 128)  # couche cachée
        self.fc2 = nn.Linear(128, 10)    # couche de sortie (logits)

    def forward(self, x):
        # Aplatir de manière robuste en préservant la dimension batch (dim 0) :
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                    # NE PAS appliquer Softmax ici
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

EPOCHS = 10
for epoch in range(EPOCHS):
    model.train() # Passe le modèle en mode entraînement
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for inputs, labels in trainloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 1. Réinitialiser les gradients
        optimizer.zero_grad(set_to_none=True)

        # 2. Passe avant (forward)
        outputs = model(inputs)

        # 3. Calcul de la perte
        loss = criterion(outputs, labels)

        # 4. Passe arrière (Rétropropagation)
        loss.backward()

        # 5. Mise à jour des poids
        optimizer.step()

        # Statistiques
        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        running_total   += labels.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc  = running_correct / running_total
    print(f"Epoch {epoch+1:02d} | loss={epoch_loss:.4f} | acc={epoch_acc:.4f}")

model.eval() # Mode évaluation
classes = trainset.classes

total = 0
correct = 0

with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

acc = correct / total
print(f"Test accuracy: {acc:.3f}")

# Sauvegarde des poids
torch.save(model.state_dict(), "mlp_model.pth")

# Exemple de chargement (sur CPU) - Vous pouvez tester cela dans un script séparé
# model2 = MLP().to("cpu")
# state = torch.load("mlp_model.pth", map_location="cpu", weights_only=True)
# model2.load_state_dict(state)
# model2.eval()