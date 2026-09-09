"""
Unit tests for modular ML subpackages:
- ml.preprocessing
- ml.training
- ml.inference
- ml.evaluation
"""
import pytest
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString

from ml.preprocessing import (
    FeatureMatrixBuilder,
    calculate_scale_pos_weight,
    compute_sample_weights,
    generate_real_labels,
    generate_synthetic_scenario_labels,
    DataLeakageAuditor,
    DataLeakageError,
)
from ml.training import (
    FloodRiskTrainer,
    SpatialZoneSplitter,
    TemporalSplitter,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_FILENAME,
)
from ml.inference import (
    RoadFloodRiskScorer,
    DrainWatchAdjuster,
    ConformalRiskPredictor,
    score_to_band,
    compute_vehicle_risk_scores,
    RISK_BAND_NAMES,
)
from ml.evaluation import (
    MetricSummary,
    find_optimal_threshold_fbeta,
    calculate_metrics,
    EvaluationReporter,
)


@pytest.fixture
def test_dataset():
    """Create a mock multi-zone dataset for end-to-end pipeline testing."""
    np.random.seed(42)
    n = 60

    # Zones covering train (1-9), val (10-12), and test (13-15)
    zone_choices = ["ZONE_01", "ZONE_05", "ZONE_10", "ZONE_11", "ZONE_13", "ZONE_14"]
    zones = [zone_choices[i % len(zone_choices)] for i in range(n)]

    r24 = np.random.uniform(20.0, 250.0, n)
    elev = np.random.uniform(2.0, 20.0, n)
    loc_ctx = np.random.uniform(-3.0, 3.0, n)
    slope = np.random.uniform(0.1, 4.0, n)
    dist_water = np.random.uniform(20.0, 1000.0, n)
    dist_drain = np.random.uniform(10.0, 100.0, n)
    hist_freq = np.random.uniform(0.0, 0.7, n)

    df = pd.DataFrame({
        "road_id": [f"road_{i}" for i in range(n)],
        "highway": ["primary" if i % 2 == 0 else "residential" for i in range(n)],
        "zone_no": zones,
        "rainfall_15min": r24 * 0.05,
        "rainfall_1h": r24 * 0.2,
        "rainfall_3h": r24 * 0.4,
        "rainfall_6h": r24 * 0.65,
        "rainfall_24h": r24,
        "elevation_mean": elev,
        "elevation_min": elev - 1.0,
        "elevation_max": elev + 1.0,
        "elevation_range": 2.0,
        "slope": slope,
        "local_elevation_context": loc_ctx,
        "dist_to_water_m": dist_water,
        "near_water_50m": (dist_water < 50).astype(float),
        "near_water_200m": (dist_water < 200).astype(float),
        "dist_to_drain_m": dist_drain,
        "has_adjacent_drain": 1.0,
        "crosses_water": 0.0,
        "historical_flood_count": hist_freq * 3,
        "historical_flood_frequency": hist_freq,
        "historical_inundation_presence": (hist_freq > 0.3).astype(float),
        "observed_flood_count": 0.0,
        "inferred_flood_count": 0.0,
        "historical_confidence": 0.6,
        "primary_evidence_type": ["OBSERVED" if i % 3 == 0 else "INFERRED" for i in range(n)],
        "sample_source": ["REAL" if i % 2 == 0 else "SYNTHETIC" for i in range(n)],
        "lanes": 2.0,
        "length_m": 100.0,
        "bridge": 0.0,
        "tunnel": 0.0,
        "rainfall_24h_x_elevation_context": r24 * loc_ctx,
        "rainfall_1h_x_dist_to_drain": (r24 * 0.2) * dist_drain,
        "hist_freq_x_rainfall_24h": hist_freq * r24,
    })

    # Ground truth label
    logits = (r24 / 120.0) - (elev / 12.0) + (hist_freq * 2.0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    df["road_inaccessible"] = (probs > 0.5).astype(float)

    return df


class TestPreprocessingSubpackage:
    """Tests for ml.preprocessing."""

    def test_sample_weighting_and_scale(self, test_dataset):
        weights = compute_sample_weights(test_dataset)
        assert len(weights) == len(test_dataset)
        assert np.all(weights >= 0.0)

        scale = calculate_scale_pos_weight(test_dataset["road_inaccessible"])
        assert scale > 0.0

    def test_leakage_auditor_pass(self, test_dataset):
        report = DataLeakageAuditor.audit_training_data(
            df=test_dataset,
            feature_columns=["rainfall_24h", "elevation_mean"],
        )
        assert report.passed is True


class TestTrainingSubpackage:
    """Tests for ml.training."""

    def test_spatial_split_zones(self, test_dataset):
        train_df, val_df, test_df = SpatialZoneSplitter.split_by_zone(test_dataset)

        assert not train_df.empty
        assert not val_df.empty
        assert not test_df.empty
        assert len(train_df) + len(val_df) + len(test_df) == len(test_dataset)

        # Ensure complete zone separation
        train_z = set(train_df["zone_no"].unique())
        val_z = set(val_df["zone_no"].unique())
        test_z = set(test_df["zone_no"].unique())

        assert not train_z.intersection(val_z)
        assert not train_z.intersection(test_z)
        assert not val_z.intersection(test_z)

    def test_temporal_split(self):
        df = pd.DataFrame({
            "event_id": ["CHENNAI_2015", "CHENNAI_2015", "CHENNAI_2021", "MICHAUNG_2023"],
            "feat": [1, 2, 3, 4],
        })
        tr, va, te = TemporalSplitter.split_by_event(df)
        assert len(tr) == 2
        assert len(va) == 1
        assert len(te) == 1

    def test_train_and_save_to_models_dir(self, test_dataset, tmp_path):
        train_df, val_df, _ = SpatialZoneSplitter.split_by_zone(test_dataset)

        feature_cols = [
            "rainfall_15min", "rainfall_1h", "rainfall_3h", "rainfall_6h", "rainfall_24h",
            "elevation_mean", "elevation_min", "elevation_max", "elevation_range",
            "slope", "local_elevation_context", "dist_to_water_m", "dist_to_drain_m",
            "historical_flood_frequency", "rainfall_24h_x_elevation_context",
            "rainfall_1h_x_dist_to_drain", "hist_freq_x_rainfall_24h"
        ]

        X_tr = train_df[feature_cols]
        y_tr = train_df["road_inaccessible"].values
        X_v = val_df[feature_cols]
        y_v = val_df["road_inaccessible"].values

        trainer = FloodRiskTrainer(n_estimators=15, max_depth=3, random_state=42)
        report = trainer.train(X_tr, y_tr, X_val=X_v, y_val=y_v)

        assert trainer.primary_model is not None
        assert "n_train" in report

        # Test serialization to specific path
        model_file = tmp_path / "test_model.pkl"
        saved_path = trainer.save(model_file)
        assert saved_path.exists()

        # Test loading
        loaded_trainer = FloodRiskTrainer().load(model_file)
        preds = loaded_trainer.predict_proba(X_v)
        assert len(preds) == len(X_v)
        assert np.all((preds >= 0.0) & (preds <= 1.0))


class TestInferenceSubpackage:
    """Tests for ml.inference."""

    def test_risk_classes_and_bands(self):
        # Verification of required risk classes: LOW, MODERATE, HIGH, CRITICAL
        assert score_to_band(10.0) == "LOW"
        assert score_to_band(35.0) == "MODERATE"
        assert score_to_band(60.0) == "HIGH"
        assert score_to_band(85.0) == "CRITICAL"

        assert "LOW" in RISK_BAND_NAMES
        assert "MODERATE" in RISK_BAND_NAMES
        assert "HIGH" in RISK_BAND_NAMES
        assert "CRITICAL" in RISK_BAND_NAMES

    def test_vehicle_risk_scores(self):
        scores = compute_vehicle_risk_scores(55.0, "HIGH")
        # In flood conditions, motorcycle risk > car risk > bus risk
        assert scores["motorcycle"] >= scores["car"]
        assert scores["car"] >= scores["bus"]

    def test_end_to_end_scoring_network(self, test_dataset):
        # Convert test_dataset to GeoDataFrame
        lines = [LineString([(80.2 + i * 0.001, 13.0), (80.2 + i * 0.001, 13.01)]) for i in range(len(test_dataset))]
        gdf = gpd.GeoDataFrame(test_dataset, geometry=lines, crs="EPSG:4326")

        scorer = RoadFloodRiskScorer()
        scored_gdf = scorer.score_network(gdf)

        assert "flood_access_risk_score" in scored_gdf.columns
        assert "base_risk_score" in scored_gdf.columns
        assert "risk_class" in scored_gdf.columns
        assert "vehicle_scores" in scored_gdf.columns
        assert "prediction_set" in scored_gdf.columns

        # Verify scores are in 0-100 range
        scores = scored_gdf["flood_access_risk_score"].values
        assert np.all((scores >= 0.0) & (scores <= 100.0))

        # Verify classes match defined bands
        for cls in scored_gdf["risk_class"].unique():
            assert cls in ["LOW", "MODERATE", "HIGH", "CRITICAL"]


class TestEvaluationSubpackage:
    """Tests for ml.evaluation."""

    def test_evaluation_reporter_text(self, test_dataset):
        # Add mock predictions
        df = test_dataset.copy()
        df["calibrated_prob"] = np.clip(df["road_inaccessible"] * 0.7 + 0.15, 0.0, 1.0)
        df["drainwatch_available"] = [True if i % 4 == 0 else False for i in range(len(df))]

        report = EvaluationReporter.evaluate_dataset(df)
        assert "overall" in report
        assert "by_zone" in report

        summary_text = EvaluationReporter.format_text_report(report)
        assert "CHENNAI FLOOD ACCESS RISK - EVALUATION REPORT" in summary_text
        assert "AUROC" in summary_text
        assert "Calibration Error" in summary_text
