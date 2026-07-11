# 3. Convert annotation to labels
import geojson, pandas as pd, numpy as np, os, glob

features = np.load("train_data/features/conch_features.npz", allow_pickle=True)
tile_coords = {os.path.basename(p): p for p in glob.glob("train_data/tiles/*_coords.csv")}

all_data = []
for geo_file in glob.glob("train_data/annotations/*.geojson"): # all file in annotations folder will read
    with open(geo_file) as f:
        geo = geojson.load(f)
    slide_name = os.path.basename(geo_file).replace(".geojson", ".ndpi")
    
    # All tiles from this slide
    coord_df = pd.read_csv(f"train_data/tiles/{slide_name[:-5]}_coords.csv")
    print(coord_df)
    tile_dict = {(row.x, row.y): row.tile_path for _, row in coord_df.iterrows()}
    
    # Positive tiles = those containing a point annotation
    positive_tiles = set()
    for feat in geo.features:
        if feat.properties.get("classification", {}).get("name") != "Vascular" and feat.properties.get("classification", {}).get("name") != "LVI":
            continue  # Skip non annotations

        geom_type = feat.geometry.type
        coords = feat.geometry.coordinates

        if geom_type == "Point":
            x, y = coords[0], coords[1]
            tile_x = (int(x) // 512) * 512
            tile_y = (int(y) // 512) * 512
            tile_path = tile_dict.get((tile_x, tile_y))
            if tile_path:
                positive_tiles.add(os.path.basename(tile_path))

        elif geom_type in ["Polygon", "MultiPolygon"]:
            points = []
            if geom_type == "Polygon":
                for ring in coords:
                    points.extend(ring)
            else:
                for poly in coords:
                    for ring in poly:
                        points.extend(ring)

            # Get unique tile positions that the polygon overlaps
            tile_positions = set()
            for pt in points:
                try:
                    x, y = float(pt[0]), float(pt[1])
                    tile_x = (int(x) // 512) * 512
                    tile_y = (int(y) // 512) * 512
                    tile_positions.add((tile_x, tile_y))
                except (TypeError, ValueError, IndexError):
                    continue  # Skip invalid points

            # Add all overlapping tiles as positive
            for tx, ty in tile_positions:
                tile_path = tile_dict.get((tx, ty))
                if tile_path:
                    positive_tiles.add(os.path.basename(tile_path))

        elif geom_type == "LineString":
            # Similar to Polygon: sample points along the line
            for pt in coords:
                try:
                    x, y = float(pt[0]), float(pt[1])
                    tile_x = (int(x) // 512) * 512
                    tile_y = (int(y) // 512) * 512
                    tile_path = tile_dict.get((tile_x, tile_y))
                    if tile_path:
                        positive_tiles.add(os.path.basename(tile_path))
                except (TypeError, ValueError, IndexError):
                    continue

    # Create labels
    for tile_name in coord_df.tile_path:
        tile_base = os.path.basename(tile_name)
        label = 1 if tile_base in positive_tiles else 0
        if tile_base not in features: continue
        all_data.append({
            "feature": features[tile_base],
            "label": label,
            "slide": slide_name,
            "tile": tile_base
        })

import pickle
with open("train_data/vessel_dataset.pkl", "wb") as f:
    pickle.dump(all_data, f)
print(f"Dataset ready: {len(all_data)} tiles ({sum(x['label'] for x in all_data)} positive)") 