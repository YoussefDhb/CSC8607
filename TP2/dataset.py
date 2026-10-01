import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

class CardioDataset(Dataset):
    def __init__(self, csv_path):
        # 1. Chargement et nettoyage
        df = pd.read_csv(csv_path, sep=";")
        df = df.drop_duplicates().drop("id", axis=1)
        
        # 2. One-hot encoding des variables catégorielles
        categorical_features = ["gender", "cholesterol", "gluc"]
        for feature in categorical_features:
            one_hot = pd.get_dummies(df[feature], dtype=np.float32)
            one_hot.columns = [f"{feature}_{c}" for c in one_hot.columns]
            df = df.drop(feature, axis=1)
            df = pd.concat([df, one_hot], axis=1)
            
        features_names = list(df.columns)
        features_names.remove("cardio")
        
        # 3. Normalisation (StandardScaler)
        self.features = df[features_names]
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features).astype(np.float32)
        
        # 4. Labels
        self.labels = df["cardio"].to_numpy().astype(np.float32).reshape(-1, 1)

    def __len__(self):
        # Retourne la taille du dataset
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Récupérer les features et le label à l'index 'idx'
        local_features = self.features[idx]
        local_labels = self.labels[idx]
        
        return {"features": local_features, "labels": local_labels}

# Test du Dataset et création des DataLoaders
if __name__ == '__main__':
    dataset = CardioDataset("data/cardio_train.csv")
    print(f"Taille totale du dataset : {len(dataset)}")

    # Découpage 80% train, 10% val, 10% test
    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)
    
    # Création des DataLoaders
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=64, shuffle=False)
    
    # Test de lecture
    batch = next(iter(train_loader))
    print(f"Shape features: {batch['features'].shape}, Shape labels: {batch['labels'].shape}")