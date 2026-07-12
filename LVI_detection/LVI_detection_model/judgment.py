import torch
import os
import pandas as pd
from PIL import Image
import numpy as np
import shutil
import argparse
import json
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer

parser = argparse.ArgumentParser(description="LVI Detection on vascular candidate tiles")
parser.add_argument(
    "--scan_folder", 
    type=str, 
    required=False,
    default="../vascular_detection_model/detection_data/",
    help="Path to the vascular candidate folder (timestamp folder) containing CSV and tiles"
)

args = parser.parse_args()

global log_file
log_file = os.path.join(args.scan_folder, "log.txt")

def log_print(msg):
    """Print to terminal and append to log file if provided"""
    print(msg)
    if log_file:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LVI_TILE_THRESHOLD = 0.50           # Tiles with prob > 0.5 are potential LVI
SLIDE_LVI_PERCENT_THRESHOLD = 0.25  # > 25% LVI tiles → slide positive

folder_path = os.path.abspath(args.scan_folder)
vascular_candidates_dir = os.path.join(folder_path, "vascular_candidates")

if not os.path.exists(folder_path):
    log_print(f"Error: Folder not found: {folder_path}")
    raise FileNotFoundError(f"Folder not found: {folder_path}")

suspicious_folder = os.path.join(vascular_candidates_dir, "lvi_suspicious")
os.makedirs(suspicious_folder, exist_ok=True)

log_print(f"Processing folder: {vascular_candidates_dir}\n")

log_print("Loading CONCH model...")
model, preprocess = create_model_from_pretrained(
    'conch_ViT-B-16',
    "hf_hub:MahmoodLab/conch",
    hf_auth_token="token.txt"
)
model.to(device)
model.eval()

tokenizer = get_tokenizer()
classes = ['Lymphovascular invasion', 'Normal Vascular']
prompts = ['an H&E image of lymphovascular invasion', 
           'an H&E image of normal vascular']

tokenized_prompts = tokenize(texts=prompts, tokenizer=tokenizer).to(device)

with torch.inference_mode():
    text_embeddings = model.encode_text(tokenized_prompts)

csv_path = None
for f in os.listdir(folder_path):
    if f.lower().endswith('.csv'):
        csv_path = os.path.join(folder_path, f)
        break

if not csv_path:
    raise FileNotFoundError("No CSV file found in the folder")

df = pd.read_csv(csv_path)
df['x'] = df['x'].astype(int)
df['y'] = df['y'].astype(int)

total_vascular_tiles = len(df)
log_print(f"Found {total_vascular_tiles} vascular candidate tiles.\n")

results = []
lvi_tile_count = 0

log_print("Running LVI classification on tiles...\n")

with torch.inference_mode():
    for _, row in df.iterrows():
        tile_filename = row['tile_filename'].strip()
        image_path = os.path.join(vascular_candidates_dir, tile_filename)

        if not os.path.isfile(image_path):
            log_print(f"Warning: Missing tile {tile_filename}")
            continue

        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = preprocess(image).unsqueeze(0).to(device)

            image_emb = model.encode_image(image_tensor, proj_contrast=True, normalize=True)

            logit_scale = model.logit_scale.exp()
            sim = (image_emb @ text_embeddings.T * logit_scale).softmax(dim=-1)
            probs = sim[0].cpu().numpy()

            prob_lvi = float(probs[0])
            pred_class = classes[probs.argmax()]

            is_lvi_tile = prob_lvi > LVI_TILE_THRESHOLD

            print(f"Tile: {tile_filename} | Vessel: {row.get('prob_vessel', 0):.4f} | "
                  f"LVI: {prob_lvi:.4f} | {pred_class} → {'LVI' if is_lvi_tile else 'Normal'}")

            result_row = row.to_dict()
            result_row.update({
                'prob_LVI': prob_lvi, 
                'predicted_class': pred_class,
                'is_lvi_tile': is_lvi_tile
            })
            results.append(result_row)

            if is_lvi_tile:
                shutil.copy2(image_path, os.path.join(suspicious_folder, tile_filename))
                lvi_tile_count += 1

        except Exception as e:
            log_print(f"Error on {tile_filename}: {e}")

lvi_percentage = lvi_tile_count / total_vascular_tiles if total_vascular_tiles > 0 else 0
slide_is_lvi_positive = lvi_percentage > SLIDE_LVI_PERCENT_THRESHOLD

log_print("\n" + "="*60)
log_print("FINAL SLIDE-LEVEL RESULT:")
log_print(f"   Total Vascular Tiles : {total_vascular_tiles}")
log_print(f"   LVI Tiles Detected   : {lvi_tile_count} ({lvi_percentage:.1%})")
log_print(f"   Slide LVI Positive   : {slide_is_lvi_positive}")
log_print("="*60 + "\n")

result_df = pd.DataFrame(results)
output_csv = os.path.join(folder_path, "lvi_candidates_with_conch_LVI.csv")
result_df.to_csv(output_csv, index=False)

result_json = os.path.join(folder_path, "lvi_result.json")
with open(result_json, "w", encoding="utf-8") as f:
    json.dump({
        "slide_is_lvi_positive": slide_is_lvi_positive,
        "lvi_tile_count": lvi_tile_count,
        "total_vascular_tiles": total_vascular_tiles
    }, f, indent=2)

log_print(f"Results saved to: {output_csv}")
log_print(f"Suspicious LVI tiles copied: {lvi_tile_count}")
log_print("LVI detection completed!")
