import torch
import torch.nn as nn
import openslide
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from conch.open_clip_custom import create_model_from_pretrained
from PIL import Image
import pandas as pd
import os
import argparse
from datetime import datetime

TILE_SIZE = 512
LVI_THRESHOLD = 0.7

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


def infer_lvi_heatmap_transfer(
    ndpi_path,
    vascular_csv_path,
    model_path=None,
    output_dir=None
):
    if not os.path.exists(ndpi_path):
        raise FileNotFoundError(f"Slide not found: {ndpi_path}")
    if not os.path.exists(vascular_csv_path):
        raise FileNotFoundError(f"Vascular CSV not found: {vascular_csv_path}")

    if output_dir is None:
        output_dir = os.path.dirname(vascular_csv_path)
    
    os.makedirs(output_dir, exist_ok=True)

    slide_name = os.path.splitext(os.path.basename(ndpi_path))[0]

    print(f"Processing: {slide_name}")
    print(f"CSV path   : {vascular_csv_path}")

    # Load models
    conch_model, preprocess = create_model_from_pretrained(
        'conch_ViT-B-16', "hf_hub:MahmoodLab/conch",
        hf_auth_token="token.txt"
    )
    conch_model.eval().cuda()

    if model_path is None or not os.path.exists(model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "../../vascular_detection_model/train_data/models/lvi_classifier_conch.pth")

    lvi_model = LVIClassifier().cuda()
    lvi_model.load_state_dict(torch.load(model_path, map_location='cuda'))
    lvi_model.eval()

    # Load CSV
    df_vasc = pd.read_csv(vascular_csv_path)
    print(f"Loaded {len(df_vasc)} candidate tiles from CSV")

    lvi_candidates = []
    csv_dir = os.path.dirname(vascular_csv_path)
    tile_dir = os.path.join(csv_dir, "vascular_candidates")

    print(f"Looking for tiles in: {tile_dir}")

    with torch.no_grad():
        for _, row in tqdm(df_vasc.iterrows(), total=len(df_vasc)):
            tile_filename = str(row.get('tile_filename', '')).strip()
            if not tile_filename:
                continue

            possible_paths = [
                os.path.join(tile_dir, tile_filename),
                os.path.join(csv_dir, tile_filename),
                os.path.join(csv_dir, "tiles", tile_filename),
            ]

            image_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    image_path = p
                    break

            if not image_path:
                print(f"❌ Tile not found: {tile_filename}")
                continue

            try:
                tile = Image.open(image_path).convert("RGB")
                input_tensor = preprocess(tile).unsqueeze(0).cuda()
                feature = conch_model.encode_image(input_tensor, proj_contrast=False, normalize=False)
                prob = torch.sigmoid(lvi_model(feature)).cpu().item()

                result_row = row.to_dict()
                result_row.update({
                    'prob_LVI': prob,
                    'predicted_class': "Lymphovascular invasion" if prob >= LVI_THRESHOLD else "Normal Vascular"
                })
                lvi_candidates.append(result_row)

            except Exception as e:
                print(f"Error processing {tile_filename}: {e}")

    print(f"\nSuccessfully processed {len(lvi_candidates)} / {len(df_vasc)} tiles")

    # Save CSV
    result_df = pd.DataFrame(lvi_candidates)
    output_csv = os.path.join(output_dir, "lvi_candidates_with_infer_LVI.csv")
    result_df.to_csv(output_csv, index=False)
    print(f"CSV saved: {output_csv} ({len(result_df)} rows)")

    if len(lvi_candidates) == 0:
        print("No tiles processed. Check the 'Tile not found' messages above.")
        return output_dir

    print("Generating LVI Heatmaps...")

    slide = openslide.OpenSlide(ndpi_path)
    width, height = slide.dimensions

    grid_h = (height + TILE_SIZE - 1) // TILE_SIZE
    grid_w = (width + TILE_SIZE - 1) // TILE_SIZE
    
    full_prob_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    threshold_grid = np.zeros((grid_h, grid_w), dtype=np.float32)

    for _, row in result_df.iterrows():
        grid_x = int(row['x']) // TILE_SIZE
        grid_y = int(row['y']) // TILE_SIZE
        prob = float(row['prob_LVI'])
        full_prob_grid[grid_y, grid_x] = prob
        if prob >= LVI_THRESHOLD:
            threshold_grid[grid_y, grid_x] = prob

    thumbnail = np.array(slide.get_thumbnail((1024, 1024)))

    plt.figure(figsize=(14, 14))
    plt.imshow(thumbnail)
    plt.imshow(full_prob_grid, cmap='hot', alpha=0.65, extent=[0, width, height, 0])
    plt.colorbar(label='LVI Probability')
    plt.title(f'{slide_name} - LVI Heatmap ')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"{slide_name}_lvi_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(14, 14))
    plt.imshow(thumbnail)
    plt.imshow(threshold_grid, cmap='hot', alpha=0.75, extent=[0, width, height, 0])
    plt.colorbar(label=f'LVI Probability (≥ {LVI_THRESHOLD})')
    plt.title(f'{slide_name} - LVI Thresholded Heatmap (≥ {LVI_THRESHOLD})')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"{slide_name}_lvi_thresholded_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Heatmaps saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised LVI Detection")
    parser.add_argument("--ndpi_path", type=str, required=True)
    parser.add_argument("--vascular_csv", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    infer_lvi_heatmap_transfer(
        ndpi_path=args.ndpi_path,
        vascular_csv_path=args.vascular_csv,
        model_path=args.model,
        output_dir=args.output
    )