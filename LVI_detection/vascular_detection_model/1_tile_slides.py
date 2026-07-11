# 1. Extract 1024×1024 tiles + background removal
import openslide
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path

TILE_SIZE = 1024
MAG_LEVEL = 0
THRESH = 210  # background threshold

script_directory = Path(__file__).parent.resolve()
os.chdir(script_directory)

os.makedirs("train_data/tiles", exist_ok=True)

for ndpi_file in os.listdir("train_data/ndpiTrain"):
    if not ndpi_file.endswith(".ndpi"): continue
    slide = openslide.OpenSlide(os.path.join("train_data/ndpiTrain", ndpi_file))
    w, h = slide.dimensions
    
    tiles_saved = 0
    coords = []
    
    for x in tqdm(range(0, w, TILE_SIZE), desc=ndpi_file):
        for y in range(0, h, TILE_SIZE):
            tile = slide.read_region((x, y), MAG_LEVEL, (TILE_SIZE, TILE_SIZE))
            tile = tile.convert("RGB")
            arr = np.array(tile)
            if np.mean(arr) < THRESH:  # skip white background
                continue
            fname = f"train_data/tiles/{ndpi_file[:-5]}_{x}_{y}.jpg"
            Image.fromarray(arr).save(fname)
            coords.append({"file": ndpi_file, "x": x, "y": y, "tile_path": fname})
            tiles_saved += 1
    
    # Save coord list
    import pandas as pd
    pd.DataFrame(coords).to_csv(f"train_data/tiles/{ndpi_file[:-5]}_coords.csv", index=False)
    print(f"{ndpi_file} → {tiles_saved} tiles")