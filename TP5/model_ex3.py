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


import torch
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(AttentionModule, self).__init__()
        # Couche pour calculer le score d'attention
        self.attention_layer = nn.Conv2d(feature_dim + hidden_dim, 1, 1)

    def forward(self, features, hidden_state):
        # features shape: (batch_size, 2048, 7, 7)
        # hidden_state shape: (batch_size, hidden_dim)

        # On "étale" l'état caché sur la grille spatiale 7x7
        hidden_state_expanded = hidden_state.unsqueeze(2).unsqueeze(3).repeat(1, 1, features.size(2), features.size(3))

        # Concaténation des caractéristiques de l'image et de l'état du RNN (sur la dimension des canaux)
        concat_features = torch.cat((features, hidden_state_expanded), dim=1)

        # Calcul des scores (batch_size, 1, 7, 7) -> on retire la dimension de taille 1
        attention_scores = self.attention_layer(concat_features).squeeze(1)

        # Application du Softmax sur les 49 régions (7x7)
        attention_weights = F.softmax(attention_scores.view(attention_scores.size(0), -1), dim=1)
        attention_weights = attention_weights.view(attention_scores.size(0), 1, attention_scores.size(1), attention_scores.size(2))

        # Calcul du vecteur de contexte par somme pondérée sur les dimensions H et W (dim 2 et 3)
        context_vector = torch.sum(torch.sum(features * attention_weights, dim=3), dim=2)

        return context_vector, attention_weights


import math

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, attention_module, attention_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_module = attention_module

        # Dimensions d'entrée des portes : input(mot) + hidden_prev + context(attention)
        total_dim = input_size + hidden_size + attention_size

        self.lin_i = nn.Linear(total_dim, hidden_size) # Input gate
        self.lin_f = nn.Linear(total_dim, hidden_size) # Forget gate
        self.lin_c = nn.Linear(total_dim, hidden_size) # Cell gate (candidat)
        self.lin_o = nn.Linear(total_dim, hidden_size) # Output gate
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, features, init_states=None):
        # x : (batch_size, sequence_size, input_size)
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t = torch.zeros(bs, self.hidden_size).to(x.device)
            c_t = torch.zeros(bs, self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states

        for t in range(seq_sz):
            x_t = x[:, t, :]

            # Calcul du vecteur d'attention contextuel
            att_t, _ = self.attention_module(features, h_t)

            # Concaténation de l'attention, du mot actuel, et de l'état caché précédent
            concat = torch.cat((att_t, x_t, h_t), dim=1)

            # Équations du LSTM
            i_t = torch.sigmoid(self.lin_i(concat))
            f_t = torch.sigmoid(self.lin_f(concat))
            g_t = torch.tanh(self.lin_c(concat))
            o_t = torch.sigmoid(self.lin_o(concat))

            # Mise à jour de l'état mémoire et de l'état caché
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(1))

        hidden_seq = torch.cat(hidden_seq, dim=1)
        return hidden_seq, (h_t, c_t)


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


def check_attention(batch_size=4, hidden_dim=512, emb_dim=300, seq_len=19):
    """vérifier les dimensions du module d'attention et du LSTM avec attention sur des tenseurs aléatoires."""
    torch.manual_seed(0)
    features = torch.randn(batch_size, 2048, 7, 7)
    hidden = torch.randn(batch_size, hidden_dim)

    attention = AttentionModule(feature_dim=2048, hidden_dim=hidden_dim)
    context_vector, attention_weights = attention(features, hidden)
    print(f"features : {tuple(features.shape)} | hidden_state : {tuple(hidden.shape)}")
    print(f"context_vector : {tuple(context_vector.shape)} | attention_weights : {tuple(attention_weights.shape)}")
    print(f"Somme des poids d'attention par image : {attention_weights.sum(dim=(1, 2, 3)).tolist()}")

    lstm = LSTMWithAttention(emb_dim, hidden_dim, attention, 2048)
    x = torch.randn(batch_size, seq_len, emb_dim)
    hidden_seq, (h_t, c_t) = lstm(x, features)
    print(f"x : {tuple(x.shape)} -> hidden_seq : {tuple(hidden_seq.shape)} | h_t : {tuple(h_t.shape)} | c_t : {tuple(c_t.shape)}")
    print(f"Paramètres : attention = {sum(p.numel() for p in attention.parameters()):,} | LSTMWithAttention (attention comprise) = {sum(p.numel() for p in lstm.parameters()):,}")


if __name__ == "__main__":
    check_attention()