# 2. Create CONCH embeddings
import torch
from conch.open_clip_custom import create_model_from_pretrained
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os, glob, numpy as np

model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token="token.txt")
model.eval().cuda()

class TileDataset(Dataset):
    def __init__(self, tile_paths):
        self.paths = tile_paths # list of file paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return preprocess(img), self.paths[i]

def main():
    all_tiles = glob.glob("train_data/tiles/*.jpg")
    dataset = TileDataset(all_tiles)
    loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)

    features = {}
    with torch.no_grad():
        for batch, paths in tqdm(loader, desc="CONCH features"):
            batch = batch.cuda()
            feat = model.encode_image(batch, proj_contrast=False, normalize=False)
            feat = feat.cpu().numpy()
            for p, f in zip(paths, feat):
                key = os.path.basename(p)
                features[key] = f

    np.savez_compressed("train_data/features/conch_features.npz", **features)
    print("Saved CONCH features:", len(features))

if __name__ == '__main__':
    main()