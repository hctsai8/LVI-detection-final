# 4. Train vascular model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import os

# Simple classifier on 512-dim CONCH features
class VesselClassifier(nn.Module):
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


class VesselDataset(Dataset):
    def __init__(self, data):
        positives = [d for d in data if d["label"] == 1]
        negatives = [d for d in data if d["label"] == 0]
        negatives = negatives[:len(positives) * 10]
        self.samples = positives + negatives

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        feat = self.samples[i]["feature"].astype(np.float32)  # shape (512,)
        label = self.samples[i]["label"]
        return torch.from_numpy(feat), torch.tensor(label).float()


def main():
    os.makedirs("train_data/models", exist_ok=True)

    dataset_path = "train_data/vessel_dataset.pkl"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Run script 3 first.")

    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded {len(data)} tiles ({sum(d['label'] for d in data)} positive vessel tiles)")

    dataset = VesselDataset(data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    clf = VesselClassifier().cuda()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8.0).cuda())
    optimizer = optim.AdamW(clf.parameters(), lr=1e-4)

    print("Starting training...")
    for epoch in range(40):
        clf.train()
        losses = []
        for feats, labels in dataloader:
            feats = feats.cuda()
            labels = labels.cuda()

            preds = clf(feats)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch+1:02d}/40 - Loss: {np.mean(losses):.5f}")

    # Save the trained classifier
    model_path = "train_data/models/vessel_classifier_conch.pth"
    torch.save(clf.state_dict(), model_path)
    print(f"\nTraining complete! Model saved to: {model_path}")


if __name__ == "__main__":
    main()