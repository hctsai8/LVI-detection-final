# 4. Train LVI model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import os
from pathlib import Path

TILE_SIZE = 512
DATA_DIR = "../vascular_detection_model/train_data"

class LVIClassifier(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.head(x).squeeze(1)


class LVIDataset(Dataset):
    def __init__(self, data):
        positives = [d for d in data if d["label"] == 1]
        negatives = [d for d in data if d["label"] == 0]
        
        negatives = negatives[:len(positives) * 8]
        self.samples = positives + negatives

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        feat = self.samples[i]["feature"].astype(np.float32)
        label = self.samples[i]["label"]
        return torch.from_numpy(feat), torch.tensor(label).float()


def main():
    os.makedirs(f"{DATA_DIR}/models", exist_ok=True)

    # Load dataset created from annotations
    dataset_path = f"{DATA_DIR}/lvi_dataset.pkl"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Run label generation first.")

    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded {len(data)} tiles ({sum(d['label'] for d in data)} positive LVI tiles)")

    dataset = LVIDataset(data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    model = LVIClassifier().cuda()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0).cuda())  # Higher weight for rare LVI
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    print("Starting LVI Model Training...")
    for epoch in range(50):  # More epochs than vascular as LVI is rarer
        model.train()
        losses = []
        for feats, labels in dataloader:
            feats = feats.cuda()
            labels = labels.cuda()

            preds = model(feats)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch+1:02d}/50 - Loss: {np.mean(losses):.5f}")

    # Save model
    model_path = f"{DATA_DIR}/models/lvi_classifier_conch.pth"
    torch.save(model.state_dict(), model_path)
    print(f"LVI Model training completed! Saved to: {model_path}")


if __name__ == "__main__":
    main()