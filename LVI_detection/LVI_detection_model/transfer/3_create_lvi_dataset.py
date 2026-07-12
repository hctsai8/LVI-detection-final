# 3. Convert LVI annotation to labels
import geojson
import pandas as pd
import numpy as np
import os
import glob
import pickle

DATA_DIR = "../vascular_detection_model/train_data"

features = np.load(f"{DATA_DIR}/features/conch_features.npz", allow_pickle=True)

all_data = []

for geo_file in glob.glob(f"{DATA_DIR}/annotations/*.geojson"):
    with open(geo_file) as f:
        geo = geojson.load(f)
    
    slide_name = os.path.basename(geo_file).replace(".geojson", ".ndpi")
    coord_df = pd.read_csv(f"{DATA_DIR}/tiles/{slide_name[:-5]}_coords.csv")
    tile_dict = {(row.x, row.y): row.tile_path for _, row in coord_df.iterrows()}

    positive_tiles = set()

    for feat in geo.features:
        if feat.properties.get("classification", {}).get("name") == "LVI":
            # Handle different geometry types
            geom_type = feat.geometry.type
            coords = feat.geometry.coordinates

            if geom_type == "Point":
                x, y = int(coords[0]), int(coords[1])
                tx = (x // 512) * 512
                ty = (y // 512) * 512
                if (tx, ty) in tile_dict:
                    positive_tiles.add(os.path.basename(tile_dict[(tx, ty)]))

            elif geom_type in ["Polygon", "MultiPolygon"]:
                points = []
                if geom_type == "Polygon":
                    for ring in coords:
                        points.extend(ring)
                else:
                    for poly in coords:
                        for ring in poly:
                            points.extend(ring)
                
                for pt in points:
                    try:
                        x, y = int(pt[0]), int(pt[1])
                        tx = (x // 512) * 512
                        ty = (y // 512) * 512
                        if (tx, ty) in tile_dict:
                            positive_tiles.add(os.path.basename(tile_dict[(tx, ty)]))
                    except:
                        continue

    # Create labels
    for _, row in coord_df.iterrows():
        tile_base = os.path.basename(row.tile_path)
        if tile_base not in features:
            continue
        label = 1 if tile_base in positive_tiles else 0
        all_data.append({
            "feature": features[tile_base],
            "label": label,
            "slide": slide_name,
            "tile": tile_base
        })

with open(f"{DATA_DIR}/lvi_dataset.pkl", "wb") as f:
    pickle.dump(all_data, f)

print(f"LVI Dataset created: {len(all_data)} tiles ({sum(x['label'] for x in all_data)} positive LVI)")