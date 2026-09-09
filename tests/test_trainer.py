"""
Unit tests for ML Training Infrastructure and Leakage Prevention (Phase 4b).
Tests:
- Data leakage detection (target leakage, provenance missing, temporal leakage)
- Isotonic probability calibration and ECE calculation
- XGBoost training with monotonic constraints
- Spatial GroupKFold cross-validation grouped by zone_no
- Model saving and loading persistence
"""
import pytest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path

from ml.leakage_audit import DataLeakageAuditor, DataLeakageError
from ml.calibration import IsotonicProbabilityCalibrator, calculate_ece, compute_reliability_curve
from ml.trainer import FloodRiskTrainer, MONOTONIC_INCREASING_FEATURES


@pytest.fixture
def synthetic_training_data():
    """Generate a clean synthetic dataset for training tests."""
    np.random.seed(42)
    n = 60
    
    # 4 distinct zones for spatial GroupKFold
    zones = ["ZONE_01", "ZONE_02", "ZONE_03", "ZONE_04"]
    zone_assign = [zones[i % 4] for i in range(n)]

    # Rainfall features
    rain_24h = np.random.uniform(10.0, 250.0, n)
    rain_1h = rain_24h * 0.25
    rain_15m = rain_1h * 0.3
    rain_3h = rain_24h * 0.4
    rain_6h = rain_24h * 0.65

    # Terrain features
    elev_mean = np.random.uniform(2.0, 25.0, n)
    elev_context = np.random.uniform(-3.0, 3.0, n)
    slope = np.random.uniform(0.1, 4.0, n)
    dist_water = np.random.uniform(10.0, 2000.0, n)
    hist_freq = np.random.uniform(0.0, 0.8, n)

    # Interaction
    rain_x_elev = rain_24h * elev_context
    rain_x_drain = rain_1h * 50.0
    hist_x_rain = hist_freq * rain_24h

    X = pd.DataFrame({
        "rainfall_15min": rain_15m,
        "rainfall_1h": rain_1h,
        "rainfall_3h": rain_3h,
        "rainfall_6h": rain_6h,
        "rainfall_24h": rain_24h,
        "elevation_mean": elev_mean,
        "elevation_min": elev_mean - 1.0,
        "elevation_max": elev_mean + 1.0,
        "elevation_range": 2.0,
        "slope": slope,
        "local_elevation_context": elev_context,
        "dist_to_water_m": dist_water,
        "near_water_50m": (dist_water < 50).astype(float),
        "near_water_200m": (dist_water < 200).astype(float),
        "dist_to_drain_m": 50.0,
        "has_adjacent_drain": 1.0,
        "crosses_water": 0.0,
        "historical_flood_count": hist_freq * 3,
        "historical_flood_frequency": hist_freq,
        "historical_inundation_presence": (hist_freq > 0.3).astype(float),
        "observed_flood_count": 0.0,
        "inferred_flood_count": 0.0,
        "historical_confidence": 0.5,
        "lanes": 2.0,
        "length_m": 120.0,
        "bridge": 0.0,
        "tunnel": 0.0,
        "rainfall_24h_x_elevation_context": rain_x_elev,
        "rainfall_1h_x_dist_to_drain": rain_x_drain,
        "hist_freq_x_rainfall_24h": hist_x_rain,
    })

    # Ground truth labels with clear signal
    logits = (rain_24h / 100.0) - (elev_mean / 10.0) + (hist_freq * 2.0) - (elev_context * 0.5)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs > 0.5).astype(float)

    # Some absent CCTV data (NaNs)
    X["drainwatch_score"] = np.nan

    return X, y, np.array(zone_assign)


class TestLeakageAuditor:
    """Tests for DataLeakageAuditor."""

    def test_target_leakage_detected(self):
        df = pd.DataFrame({
            "feat1": [1, 2],
            "road_inaccessible": [0, 1],
            "sample_source": ["REAL", "REAL"],
        })
        with pytest.raises(DataLeakageError, match="Target variable 'road_inaccessible' is included"):
            DataLeakageAuditor.audit_training_data(
                df=df,
                feature_columns=["feat1", "road_inaccessible"],
                target_column="road_inaccessible",
            )

    def test_missing_source_detected(self):
        df = pd.DataFrame({
            "feat1": [1, 2],
            "road_inaccessible": [0, 1],
        })
        with pytest.raises(DataLeakageError, match="Provenance column 'sample_source' is missing"):
            DataLeakageAuditor.audit_training_data(
                df=df,
                feature_columns=["feat1"],
                target_column="road_inaccessible",
            )

    def test_temporal_leakage_detected(self):
        event_time = datetime(2023, 12, 4, 6, 0, 0)
        df = pd.DataFrame({
            "feat1": [1, 2],
            "obs_time": [datetime(2023, 12, 4, 5, 0, 0), datetime(2023, 12, 4, 10, 0, 0)],
            "road_inaccessible": [0, 1],
            "sample_source": ["REAL", "REAL"],
        })
        with pytest.raises(DataLeakageError, match="TEMPORAL LEAKAGE"):
            DataLeakageAuditor.audit_training_data(
                df=df,
                feature_columns=["feat1"],
                event_timestamp=event_time,
                eval_timestamp_col="obs_time",
            )

    def test_clean_data_passes_audit(self):
        df = pd.DataFrame({
            "feat1": [1, 2],
            "road_inaccessible": [0, 1],
            "sample_source": ["REAL", "SYNTHETIC"],
            "zone_no": ["ZONE_01", "ZONE_02"],
        })
        report = DataLeakageAuditor.audit_training_data(
            df=df,
            feature_columns=["feat1"],
        )
        assert report.passed is True
        assert len(report.issues) == 0


class TestCalibration:
    """Tests for calibration and ECE calculation."""

    def test_ece_perfect_calibration(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        ece = calculate_ece(y_true, y_prob)
        assert ece == pytest.approx(0.0)

    def test_isotonic_calibrator_fitting(self):
        # Raw probabilities that are systematically under-confident
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        raw_probs = np.linspace(0.1, 0.6, 10)

        calibrator = IsotonicProbabilityCalibrator()
        calibrator.fit(raw_probs, y_true)
        assert calibrator.is_fitted

        cal_probs = calibrator.calibrate(raw_probs)
        assert len(cal_probs) == 10
        assert np.all(cal_probs >= 0.0) and np.all(cal_probs <= 1.0)

        report = calibrator.evaluate_calibration(raw_probs, y_true)
        assert "calibrated_ece" in report
        assert "overall_curve" in report


class TestTrainer:
    """Tests for FloodRiskTrainer."""

    def test_trainer_fit_and_monotonicity(self, synthetic_training_data):
        X, y, zones = synthetic_training_data

        trainer = FloodRiskTrainer(
            n_estimators=30,
            max_depth=3,
            use_monotonic_constraints=True,
            use_ensemble=False,
            random_state=42,
        )
        report = trainer.train(X, y)

        assert trainer.primary_model is not None
        assert "n_train" in report
        assert report["n_train"] == len(X)

        # Test Monotonicity: Increasing rainfall must NEVER decrease predicted risk
        sample_row = X.iloc[[0]].copy()
        
        # Test across increasing 24h rainfall levels
        rainfall_steps = [25.0, 50.0, 100.0, 150.0, 200.0, 300.0]
        preds = []
        for r in rainfall_steps:
            sample_row["rainfall_24h"] = r
            sample_row["rainfall_6h"] = r * 0.65
            sample_row["rainfall_3h"] = r * 0.4
            sample_row["rainfall_1h"] = r * 0.25
            sample_row["rainfall_15min"] = r * 0.1
            sample_row["hist_freq_x_rainfall_24h"] = sample_row["historical_flood_frequency"] * r
            sample_row["rainfall_24h_x_elevation_context"] = r * sample_row["local_elevation_context"]
            sample_row["rainfall_1h_x_dist_to_drain"] = (r * 0.25) * sample_row["dist_to_drain_m"]
            p = trainer.predict_proba(sample_row)[0]
            preds.append(p)

        # Monotonicity check: preds[i+1] >= preds[i] - epsilon (float precision tolerance)
        for i in range(len(preds) - 1):
            assert preds[i + 1] >= preds[i] - 1e-4

    def test_spatial_cross_validation(self, synthetic_training_data):
        X, y, zones = synthetic_training_data

        trainer = FloodRiskTrainer(
            n_estimators=20,
            max_depth=3,
            use_ensemble=False,
        )
        fold_results = trainer.spatial_cross_validate(X, y, zones=zones, n_splits=3)

        assert len(fold_results) == 3
        for fold in fold_results:
            assert "val_auroc" in fold
            assert "val_auprc" in fold
            assert "val_ece" in fold

    def test_save_and_load(self, synthetic_training_data, tmp_path):
        X, y, _ = synthetic_training_data

        trainer = FloodRiskTrainer(n_estimators=15, max_depth=2, use_ensemble=False)
        trainer.train(X, y)
        orig_preds = trainer.predict_proba(X.iloc[:5])

        save_path = tmp_path / "model.pkl"
        trainer.save(save_path)
        assert save_path.exists()

        loaded_trainer = FloodRiskTrainer()
        loaded_trainer.load(save_path)

        loaded_preds = loaded_trainer.predict_proba(X.iloc[:5])
        np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-5)
