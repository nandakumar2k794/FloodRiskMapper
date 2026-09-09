"""
Script: Train and serialize the Chennai Flood Access Risk Model.
Saves model bundle to models/flood_risk_xgboost.pkl.
"""
from pathlib import Path
import sys

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString

from ml.preprocessing import (
    FeatureMatrixBuilder,
    compute_sample_weights,
    generate_synthetic_scenario_labels,
    DataLeakageAuditor,
    MODEL_FEATURE_NAMES,
)
from ml.training import (
    FloodRiskTrainer,
    SpatialZoneSplitter,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_FILENAME,
)
from ml.evaluation import EvaluationReporter


def generate_chennai_training_dataset(n_segments: int = 150) -> pd.DataFrame:
    """
    Generate realistic multi-zone Chennai road feature training samples
    covering all 15 GCC zones, various highway types, and rainfall scenarios.
    """
    np.random.seed(42)

    # 15 GCC Administrative Zones
    all_zones = [f"ZONE_{i:02d}" for i in range(1, 16)]
    zones = [all_zones[i % len(all_zones)] for i in range(n_segments)]

    highways = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential"]
    hw_types = [highways[i % len(highways)] for i in range(n_segments)]

    # Dynamic rainfall scenarios (50mm to 300mm)
    r24 = np.random.uniform(30.0, 280.0, n_segments)
    r1h = r24 * np.random.uniform(0.15, 0.35, n_segments)
    r15m = r1h * 0.3
    r3h = r24 * 0.45
    r6h = r24 * 0.70

    # Topography: Coastal lowlands (1m to 25m)
    elev_mean = np.random.uniform(2.0, 22.0, n_segments)
    loc_ctx = np.random.uniform(-3.5, 3.5, n_segments)
    slope = np.random.uniform(0.1, 3.5, n_segments)

    # Proximity
    dist_water = np.random.uniform(15.0, 3000.0, n_segments)
    dist_drain = np.random.uniform(5.0, 150.0, n_segments)
    hist_freq = np.random.uniform(0.0, 0.75, n_segments)

    df = pd.DataFrame({
        "road_id": [f"road_chn_{i:04d}" for i in range(n_segments)],
        "segment_id": [f"road_chn_{i:04d}" for i in range(n_segments)],
        "highway": hw_types,
        "zone_no": zones,
        "rainfall_15min": r15m,
        "rainfall_1h": r1h,
        "rainfall_3h": r3h,
        "rainfall_6h": r6h,
        "rainfall_24h": r24,
        "rainfall_source_type": "SIMULATED",
        "elevation_mean": elev_mean,
        "elevation_min": elev_mean - 0.8,
        "elevation_max": elev_mean + 0.8,
        "elevation_range": 1.6,
        "slope": slope,
        "local_elevation_context": loc_ctx,
        "dist_to_water_m": dist_water,
        "near_water_50m": (dist_water < 50.0).astype(float),
        "near_water_200m": (dist_water < 200.0).astype(float),
        "dist_to_drain_m": dist_drain,
        "has_adjacent_drain": (dist_drain <= 15.0).astype(float),
        "crosses_water": (dist_water < 25.0).astype(float),
        "historical_flood_count": np.round(hist_freq * 3.0),
        "historical_flood_frequency": hist_freq,
        "historical_inundation_presence": (hist_freq > 0.3).astype(float),
        "observed_flood_count": 0.0,
        "inferred_flood_count": 0.0,
        "historical_confidence": np.random.uniform(0.5, 0.9, n_segments),
        "primary_evidence_type": ["SYNTHETIC"] * n_segments,
        "sample_source": ["SYNTHETIC"] * n_segments,
        "lanes": [4.0 if hw in ["motorway", "primary"] else 2.0 for hw in hw_types],
        "length_m": np.random.uniform(50.0, 400.0, n_segments),
        "bridge": 0.0,
        "tunnel": 0.0,
        "rainfall_24h_x_elevation_context": r24 * loc_ctx,
        "rainfall_1h_x_dist_to_drain": r1h * dist_drain,
        "hist_freq_x_rainfall_24h": hist_freq * r24,
    })

    # Physics-based scenario labeling
    labeled = generate_synthetic_scenario_labels(df, noise_flip_rate=0.08, random_seed=42)
    return labeled


def train_and_serialize_model() -> Path:
    """
    Execute training pipeline, evaluate on validation zones, and save model to models/
    """
    print("Generating Chennai multi-zone training scenarios...")
    data = generate_chennai_training_dataset(n_segments=180)

    # 1. Split by GCC zones (Train: 1-9, Val: 10-12, Test: 13-15)
    train_df, val_df, test_df = SpatialZoneSplitter.split_by_zone(data)
    print(f"Spatial Partition: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test.")

    # 2. Leakage Audit
    feature_cols = [
        c for c in MODEL_FEATURE_NAMES if c in data.columns
    ]
    audit = DataLeakageAuditor.audit_training_data(
        df=train_df,
        feature_columns=feature_cols,
        target_column="road_inaccessible",
        source_column="sample_source",
        zone_column="zone_no",
    )
    print(f"Data Leakage Audit: {'PASSED' if audit.passed else 'FAILED'}")

    # 3. Sample Weights
    sample_weights = compute_sample_weights(train_df)

    X_train = train_df[feature_cols]
    y_train = train_df["road_inaccessible"].values

    X_val = val_df[feature_cols]
    y_val = val_df["road_inaccessible"].values

    X_test = test_df[feature_cols]
    y_test = test_df["road_inaccessible"].values

    # 4. Train Model with Monotonic Constraints and Isotonic Calibration
    trainer = FloodRiskTrainer(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        use_monotonic_constraints=True,
        random_state=42,
    )

    print("Fitting physically constrained XGBoost model with isotonic calibration...")
    report = trainer.train(
        X_train=X_train,
        y_train=y_train,
        sample_weights=sample_weights,
        X_val=X_val,
        y_val=y_val,
    )

    # 5. Evaluate on Held-Out Test Zone (Zones 13-15)
    test_eval_df = test_df.copy()
    test_eval_df["calibrated_prob"] = trainer.predict_proba(X_test)
    test_eval_df["drainwatch_available"] = False

    eval_results = EvaluationReporter.evaluate_dataset(test_eval_df)
    summary_text = EvaluationReporter.format_text_report(eval_results)
    print("\n" + summary_text)

    # 6. Serialize Model to models/
    out_path = trainer.save()
    print(f"\nTrained model successfully saved to: {out_path.resolve()}")
    return out_path


if __name__ == "__main__":
    train_and_serialize_model()
