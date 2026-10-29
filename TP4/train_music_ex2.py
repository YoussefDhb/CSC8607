import os
import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Reproductibilité
torch.manual_seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "/mnt/ssd/irishman"

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

train_data = load_data(os.path.join(DATA_DIR, "train.json"))
val_data   = load_data(os.path.join(DATA_DIR, "validation.json"))

print(f"Nombre de chansons (Train) : {len(train_data)}")
print(f"Nombre de chansons (Val)   : {len(val_data)}")
print("\n--- Exemple de partition (Train[0]) ---")
print(train_data[0]["abc notation"])

# 1. Extraction déterministe (triée) des caractères uniques
all_text = "".join(song["abc notation"] for song in train_data + val_data)
unique_chars = sorted(list(set(all_text)))
vocab_size = len(unique_chars)
print(f"Taille du vocabulaire : {vocab_size} caractères uniques")

# 2. Mappings caractère <-> index
char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
index_to_char = {idx: char for idx, char in enumerate(unique_chars)}

def vectorize_string(s):
    return [char_to_index[char] for char in s]

# 3. Statistiques de longueur et Padding
lengths = [len(song["abc notation"]) for song in train_data]
max_len = max(lengths)
avg_len = sum(lengths) / len(lengths)
print(f"Longueur moyenne : {avg_len:.1f} | Longueur max : {max_len}")

SEQ_LENGTH = 500  # Longueur fixe choisie (environ 2x la longueur moyenne)

def pad_or_truncate(sequence, max_length=SEQ_LENGTH):
    if len(sequence) < max_length:
        return sequence + " " * (max_length - len(sequence))
    else:
        return sequence[:max_length]


class MusicDataset(Dataset):
    def __init__(self, raw_data, seq_length=SEQ_LENGTH, subset_size=None):
        if subset_size is not None:
            raw_data = raw_data[:subset_size]
        
        self.data = [
            vectorize_string(pad_or_truncate(song["abc notation"], seq_length))
            for song in raw_data
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        # Entrée : tous les caractères sauf le dernier
        x = torch.tensor(seq[:-1], dtype=torch.long)
        # Cible : tous les caractères sauf le premier (décalage de +1)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


class MusicModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_size=512, rnn_type="lstm"):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        elif self.rnn_type == "rnn":
            self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        else:
            raise ValueError("rnn_type doit être 'rnn' ou 'lstm'")
            
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden_state=None):
        # x : (batch_size, seq_len)
        embeds = self.embedding(x)               # -> (batch_size, seq_len, embedding_dim)
        out, hidden_state = self.rnn(embeds, hidden_state) # -> out : (batch_size, seq_len, hidden_size)
        logits = self.fc(out)                    # -> (batch_size, seq_len, vocab_size)
        return logits, hidden_state

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_dataset(batch_size=128):
    """vérifier les dimensions et le type des tenseurs renvoyés par le DataLoader."""
    dataset = MusicDataset(train_data, subset_size=1000)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    print(f"\nx : {tuple(x.shape)}, {x.dtype} | y : {tuple(y.shape)}, {y.dtype}")
    print(f"Décalage d'un pas : x[0, 1:6] == y[0, :5] -> {torch.equal(x[0, 1:6], y[0, :5])}")


def check_models():
    """afficher le nombre de paramètres de chaque couche et la structure de hidden_state pour le RNN et le LSTM."""
    x = torch.zeros(2, 10, dtype=torch.long)
    for rnn_type in ["rnn", "lstm"]:
        model = MusicModel(vocab_size, embedding_dim=256, hidden_size=512, rnn_type=rnn_type)
        print(f"\n--- {rnn_type.upper()} ---")
        for name in ["embedding", "rnn", "fc"]:
            print(f"{name:9s} : {count_parameters(getattr(model, name)):,} paramètres")
        print(f"Total     : {count_parameters(model):,} paramètres")
        logits, hidden_state = model(x)
        if isinstance(hidden_state, tuple):
            print(f"logits : {tuple(logits.shape)} | hidden_state : tuple (h_n {tuple(hidden_state[0].shape)}, c_n {tuple(hidden_state[1].shape)})")
        else:
            print(f"logits : {tuple(logits.shape)} | hidden_state : tenseur h_n {tuple(hidden_state.shape)}")


if __name__ == "__main__":
    check_dataset()
    check_models()