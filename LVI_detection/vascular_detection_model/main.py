import torch
import torch.nn as nn
import os
import openslide
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from conch.open_clip_custom import create_model_from_pretrained
from PIL import Image
import argparse
import pandas as pd
from datetime import datetime

# Vessel Classifier
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


def infer_vessel_heatmap(
    ndpi_path,
    model_path="train_data/models/vessel_classifier_conch.pth",
    output_dir="train_data/results",
    lvi_candidate_threshold=0.50,
    lvi_candidates_dir="train_data/vascular_candidates",
    save_images=True
):
    if not os.path.exists(ndpi_path):
        raise FileNotFoundError(f"Slide not found: {ndpi_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Prepare output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(lvi_candidates_dir, exist_ok=True)

    slide_name = os.path.splitext(os.path.basename(ndpi_path))[0]

    global log_file
    log_file = os.path.join(output_dir, "log.txt")

    # Load models
    print("Loading CONCH model...")
    conch_model, preprocess = create_model_from_pretrained(
        'conch_ViT-B-16',
        "hf_hub:MahmoodLab/conch",
        hf_auth_token="token.txt"
    )
    conch_model.eval().cuda()

    print("Loading trained vessel classifier...")
    clf = VesselClassifier().cuda()
    clf.load_state_dict(torch.load(model_path, map_location='cuda'))
    clf.eval()

    # Slide info
    slide = openslide.OpenSlide(ndpi_path)
    width, height = slide.dimensions
    print(f"Processing slide: {slide_name} | Size: {width}x{height}")

    grid_h = (height + 511) // 512
    grid_w = (width + 511) // 512
    full_scores = np.zeros((grid_h, grid_w), dtype=np.float32)      # All probabilities
    threshold_scores = np.zeros((grid_h, grid_w), dtype=np.float32) # Only >= threshold

    lvi_candidates = []
    tile_count = 0
    vessel_tile_count = 0

    print("Running vascular detection...")
    with torch.no_grad():
        for x in tqdm(range(0, width, 512), desc="Columns"):
            for y in range(0, height, 512):
                tile = slide.read_region((x, y), 0, (512, 512)).convert("RGB")
                tile_np = np.array(tile)

                if np.mean(tile_np) > 220:
                    continue

                tile_count += 1

                input_tensor = preprocess(tile).unsqueeze(0).cuda()
                feat = conch_model.encode_image(input_tensor, proj_contrast=False, normalize=False)
                prob = torch.sigmoid(clf(feat)).cpu().item()

                grid_y = y // 512
                grid_x = x // 512
                
                full_scores[grid_y, grid_x] = prob

                if prob >= lvi_candidate_threshold:
                    threshold_scores[grid_y, grid_x] = prob
                    vessel_tile_count += 1

                    tile_filename = f"tile_x{x:06d}_y{y:06d}_prob{prob:.3f}.png"

                    if save_images:
                        tile.save(os.path.join(lvi_candidates_dir, tile_filename))

                    lvi_candidates.append({
                        'slide': os.path.basename(ndpi_path),
                        'x': x,
                        'y': y,
                        'prob_vessel': prob,
                        'tile_filename': tile_filename if save_images else None
                    })

    print(f"Processed {tile_count} tissue tiles")
    print(f"→ {vessel_tile_count} vascular tiles (≥ {lvi_candidate_threshold})")

    # Save CSV
    if lvi_candidates:
        df = pd.DataFrame(lvi_candidates)
        csv_path = os.path.join(output_dir, "lvi_candidates.csv")
        df.to_csv(csv_path, index=False)
        print(f"LVI candidate list saved → {csv_path}")

    # Generate threshold heatmap
    print("Generating vascular heatmaps...")

    thumbnail = np.array(slide.get_thumbnail((1024, 1024)))

    plt.figure(figsize=(14, 14))
    plt.imshow(thumbnail)
    plt.imshow(threshold_scores, cmap='hot', alpha=0.75, extent=[0, width, height, 0])
    plt.colorbar(label=f'Vascular Probability (≥ {lvi_candidate_threshold})')
    plt.title(f'{slide_name} - Vascular Thresholded Heatmap (≥ {lvi_candidate_threshold})')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"{slide_name}_vascular_thresholded_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Vascular thresholded heatmap saved in: {output_dir}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Vascular Detection with Two Heatmaps")
    
    parser.add_argument("--ndpi_path", type=str, default=os.path.join(script_dir, "train_data/ndpi/C23000651_DI DI_1A_HE - 2025-04-03 13.10.36.ndpi"))
    parser.add_argument("--model", type=str, default=os.path.join(script_dir, "train_data/models/vessel_classifier_conch.pth"))
    parser.add_argument("--output", type=str, default=os.path.join(script_dir, "detection_data/results"))
    parser.add_argument("--lvi-dir", type=str, default=os.path.join(script_dir, "detection_data/vascular_candidates"))
    parser.add_argument("--lvi-th", type=float, default=0.50)
    parser.add_argument("--no-save-tiles", action="store_true")

    args = parser.parse_args()

    infer_vessel_heatmap(
        ndpi_path=args.ndpi_path,
        model_path=args.model,
        output_dir=args.output,
        lvi_candidates_dir=args.lvi_dir,
        lvi_candidate_threshold=args.lvi_th,
        save_images=not args.no_save_tiles,
    )