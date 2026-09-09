# Chennai Flood Risk Model Bundle

Standalone deployment package containing the trained XGBoost model, feature schema, inference script, and precomputed GeoJSON/GPKG layers for Chennai.

---

## 📦 Directory Contents

- `final_model.json` — Trained production XGBoost model
- `feature_columns.json` — Exact 13 features in order:
  `["rainfall", "distance_to_road", "distance_to_drainage", "distance_to_river", "drainage_density", "land_use", "soil", "mean_elevation", "min_elevation", "max_elevation", "elevation_range", "elevation_std", "slope"]`
- `predict.py` — Standalone Python script for single-point and batch inference
- `chennai_flood_risk.geojson` — 4,361 grid cells with coordinates, `flood_probability`, and `risk_class` (EPSG:4326)
- `chennai_flood_risk.gpkg` — OGC standard GeoPackage for GIS analysis
- `figures/` — Map visualizations:
  - `flood_probability_map.png`
  - `flood_risk_classes_map.png`
  - `elevation_map.png`
  - `rainfall_map.png`
  - `fi_dem.png`

---

## 🚀 Quick Usage

### 1. Single Point Inference
```bash
python predict.py --rainfall 180 --distance_to_road 20 --distance_to_drainage 50 --distance_to_river 300 --mean_elevation 6.5 --slope 0.8
```

### 2. Batch CSV Inference
```bash
python predict.py --input-csv my_features.csv --output-csv predictions.csv
```

### 3. Python Integration
```python
import json
import xgboost as xgb
import pandas as pd

with open("feature_columns.json") as f:
    features = json.load(f)

model = xgb.XGBClassifier()
model.load_model("final_model.json")

# Sample data
data = {"rainfall": 180, "distance_to_road": 20, "distance_to_drainage": 50, "distance_to_river": 300,
        "drainage_density": 10, "land_use": 1, "soil": 1, "mean_elevation": 6.5, "min_elevation": 5.0,
        "max_elevation": 7.0, "elevation_range": 2.0, "elevation_std": 0.5, "slope": 0.8}

df = pd.DataFrame([data])[features]
prob = float(model.predict_proba(df)[:, 1][0])
print("Flood Probability:", prob)
```
