"""
Retrain XGBoost with balanced, realistic features for Chennai road flood risk.
Saves new model to trained/final_model.json (overwrites the saturated one).
"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

SEED = 42
N = 2000  # training samples

rng = np.random.RandomState(SEED)

# ─── Realistic feature distributions ─────────────────────────────────────────
# distance_to_road: metres from flood grid cell to nearest road (0 = on road)
# For urban Chennai roads this is 0-50m; cells far from roads don't flood roads
dist_to_road     = rng.uniform(0, 200, N)          # 0-200m
dist_to_drainage = rng.uniform(5, 500, N)          # 5-500m
dist_to_river    = rng.uniform(20, 5000, N)        # 20-5000m
drainage_density = rng.uniform(1, 50, N)           # 1-50 km/km2
land_use         = rng.choice([1, 2, 3], N)        # 1=residential, 2=commercial, 3=wetland
soil             = rng.choice([1, 2, 3], N)        # 1=clay, 2=loam, 3=sandy
mean_elev        = rng.uniform(0.5, 20.0, N)       # 0.5-20m
slope            = rng.uniform(0.1, 10.0, N)       # 0.1-10 degrees
rainfall         = rng.uniform(10, 350, N)         # 10-350mm

min_elev  = np.maximum(0.1, mean_elev - slope * 0.5)
max_elev  = mean_elev + slope * 0.5
elev_range = np.maximum(0.1, slope)
elev_std  = np.maximum(0.1, slope * 0.3)

# ─── Physics-based flood label ────────────────────────────────────────────────
# Flood = 1 when: high rainfall AND low elevation AND near river AND poor drainage
rain_norm    = np.clip(rainfall / 300.0, 0, 1)
elev_norm    = np.clip(1.0 - mean_elev / 15.0, 0, 1)
river_norm   = np.clip(1.0 - dist_to_river / 1000.0, 0, 1)
drain_norm   = np.clip(1.0 - dist_to_drainage / 300.0, 0, 1)

flood_prob = (
    0.40 * rain_norm +
    0.25 * elev_norm +
    0.20 * river_norm +
    0.10 * drain_norm +
    0.05 * (land_use == 3).astype(float)  # wetland
)
# Add noise
flood_prob = np.clip(flood_prob + rng.normal(0, 0.05, N), 0, 1)
labels = (flood_prob > 0.45).astype(int)

print(f"Class balance — Flood=1: {labels.mean():.2%}, Flood=0: {(1-labels).mean():.2%}")

# ─── Build DataFrame ──────────────────────────────────────────────────────────
df = pd.DataFrame({
    'rainfall':            rainfall,
    'distance_to_road':    dist_to_road,
    'distance_to_drainage': dist_to_drainage,
    'distance_to_river':   dist_to_river,
    'drainage_density':    drainage_density,
    'land_use':            land_use.astype(float),
    'soil':                soil.astype(float),
    'mean_elevation':      mean_elev,
    'min_elevation':       min_elev,
    'max_elevation':       max_elev,
    'elevation_range':     elev_range,
    'elevation_std':       elev_std,
    'slope':               slope,
})
feature_cols = list(df.columns)

X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=SEED, stratify=labels)

# ─── Train XGBoost ────────────────────────────────────────────────────────────
pos = y_train.sum()
neg = (y_train == 0).sum()
spw = neg / pos

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=spw,
    eval_metric='logloss',
    random_state=SEED,
    n_jobs=1,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# ─── Evaluate ─────────────────────────────────────────────────────────────────
probs_test = model.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, probs_test)
print(f"Test ROC-AUC: {auroc:.4f}")
print(f"Prob range on test: {probs_test.min():.4f} - {probs_test.max():.4f}")
print(f"Avg prob: {probs_test.mean():.4f}")

# Quick sanity: low risk vs high risk inputs
low_risk  = pd.DataFrame([{'rainfall': 20,  'distance_to_road': 100, 'distance_to_drainage': 400, 'distance_to_river': 3000, 'drainage_density': 3,  'land_use': 1, 'soil': 1, 'mean_elevation': 14.0, 'min_elevation': 13.0, 'max_elevation': 15.0, 'elevation_range': 2.0, 'elevation_std': 0.6, 'slope': 4.0}])
high_risk = pd.DataFrame([{'rainfall': 250, 'distance_to_road': 5,   'distance_to_drainage': 10,  'distance_to_river': 30,   'drainage_density': 40, 'land_use': 3, 'soil': 1, 'mean_elevation': 1.5,  'min_elevation': 0.5,  'max_elevation': 2.5,  'elevation_range': 1.0, 'elevation_std': 0.3, 'slope': 0.2}])

p_low  = model.predict_proba(low_risk[feature_cols])[:, 1][0]
p_high = model.predict_proba(high_risk[feature_cols])[:, 1][0]
print(f"\nSanity: low-risk input  => prob={p_low:.4f}")
print(f"Sanity: high-risk input => prob={p_high:.4f}")

if p_high > p_low + 0.1:
    print("✅ Model is DISCRIMINATIVE — saving...")
    model.save_model(str(Path(__file__).parent / 'final_model.json'))
    print("✅ Saved to trained/final_model.json")
else:
    print("❌ Model still not discriminative enough. Check training labels.")
