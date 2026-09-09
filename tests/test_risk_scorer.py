"""
Unit tests for End-to-End Inference and Scoring Pipeline (Phase 4c).
Tests:
- DrainWatch Stage 2 additive adjustments with staleness decay
- Zero adjustment for roads without CCTV cameras
- Conformal prediction set generation and epistemic uncertainty indicators
- Vehicle-specific score modulation (motorcycle vs car vs bus)
- Full road network scoring pipeline
- Scenario extrapolation detection (> 300mm 24h rain)
"""
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString

from ml.drainwatch_adjustment import DrainWatchAdjuster, get_staleness_multiplier
from ml.uncertainty import ConformalRiskPredictor, score_to_band
from ml.risk_scorer import RoadFloodRiskScorer, compute_vehicle_risk_scores


@pytest.fixture
def sample_roads_gdf():
    """Create a minimal road GeoDataFrame."""
    lines = [
        LineString([(80.20, 13.00), (80.21, 13.00)]),
        LineString([(80.22, 13.02), (80.23, 13.02)]),
        LineString([(80.24, 13.04), (80.25, 13.04)]),
    ]
    return gpd.GeoDataFrame(
        {
            "segment_id": ["road_1", "road_2", "road_3"],
            "highway": ["primary", "residential", "secondary"],
            "lanes": [4, 2, 2],
            "zone_no": ["ZONE_05", "ZONE_13", "ZONE_10"],
            "elevation_mean": [6.0, 15.0, 4.0],
            "slope": [0.5, 3.0, 0.2],
            "local_elevation_context": [-1.5, 2.0, -2.2],
            "dist_to_water_m": [50.0, 400.0, 30.0],
            "dist_to_drain_m": [10.0, 100.0, 5.0],
            "historical_flood_frequency": [0.33, 0.0, 0.66],
        },
        geometry=lines,
        crs="EPSG:4326",
    )


class TestDrainWatchAdjuster:
    """Tests for DrainWatch Stage 2 additive adjustment."""

    def test_staleness_decay_multipliers(self):
        assert get_staleness_multiplier(0.5) == 1.0     # < 1 hour: 1.0x
        assert get_staleness_multiplier(2.5) == 0.7     # 1 - 4 hours: 0.7x
        assert get_staleness_multiplier(6.0) == 0.3     # 4 - 12 hours: 0.3x
        assert get_staleness_multiplier(15.0) == 0.0    # > 12 hours: 0.0x
        assert get_staleness_multiplier(None) == 0.0    # None / missing: 0.0x

    def test_camera_vs_uncamera_adjustments(self):
        adjuster = DrainWatchAdjuster()
        df = pd.DataFrame({
            "segment_id": ["r_blocked", "r_clear", "r_no_cam"],
            "base_risk_score": [40.0, 40.0, 40.0],
            "drainwatch_available": [True, True, False],
            "drainwatch_observation_class": ["drain_blocked_garbage", "drain_clear", None],
            "drainwatch_confidence": [1.0, 1.0, np.nan],
            "drainwatch_data_age_hours": [0.5, 0.5, np.nan],
        })
        res = adjuster.apply_adjustments(df)

        # r_blocked: +15 delta -> adjusted score = 55.0
        r_blk = res.loc[res["segment_id"] == "r_blocked"].iloc[0]
        assert r_blk["drainwatch_delta"] == pytest.approx(15.0)
        assert r_blk["adjusted_risk_score"] == pytest.approx(55.0)

        # r_clear: -5 delta -> adjusted score = 35.0
        r_clr = res.loc[res["segment_id"] == "r_clear"].iloc[0]
        assert r_clr["drainwatch_delta"] == pytest.approx(-5.0)
        assert r_clr["adjusted_risk_score"] == pytest.approx(35.0)

        # r_no_cam: no camera -> delta = 0.0 -> adjusted score = 40.0
        r_nc = res.loc[res["segment_id"] == "r_no_cam"].iloc[0]
        assert r_nc["drainwatch_delta"] == pytest.approx(0.0)
        assert r_nc["adjusted_risk_score"] == pytest.approx(40.0)

    def test_score_clipping_bounds(self):
        adjuster = DrainWatchAdjuster()
        df = pd.DataFrame({
            "base_risk_score": [90.0, 3.0],
            "drainwatch_available": [True, True],
            "drainwatch_observation_class": ["road_waterlogged", "drain_clear"],
            "drainwatch_confidence": [1.0, 1.0],
            "drainwatch_data_age_hours": [0.1, 0.1],
        })
        res = adjuster.apply_adjustments(df)
        # 90 + 30 = 120 -> clipped to 100.0
        assert res["adjusted_risk_score"].iloc[0] == 100.0
        # 3 - 5 = -2 -> clipped to 0.0
        assert res["adjusted_risk_score"].iloc[1] == 0.0


class TestUncertainty:
    """Tests for ConformalRiskPredictor and uncertainty metrics."""

    def test_prediction_set_generation(self):
        predictor = ConformalRiskPredictor(coverage_level=0.90)
        predictor.conformal_quantile = 0.15

        # Very low probability (0.05) -> credible interval [0.0, 0.20] -> only LOW band
        set_low = predictor.predict_prediction_set(0.05)
        assert set_low == ["LOW"]

        # Borderline probability (0.24) -> interval [0.09, 0.39] -> spans LOW and MEDIUM
        set_mid = predictor.predict_prediction_set(0.24)
        assert "LOW" in set_mid and "MEDIUM" in set_mid

    def test_enrich_with_uncertainty(self):
        predictor = ConformalRiskPredictor()
        predictor.conformal_quantile = 0.20

        df = pd.DataFrame({
            "calibrated_prob": [0.05, 0.48],
        })
        enriched = predictor.enrich_with_uncertainty(df, extrapolation=False)
        assert "prediction_set" in enriched.columns
        assert "is_uncertain" in enriched.columns
        # 0.48 has wide spread spanning multiple bands -> is_uncertain = True
        assert bool(enriched["is_uncertain"].iloc[1]) is True


class TestRiskScorer:
    """Tests for RoadFloodRiskScorer."""

    def test_vehicle_risk_modulation(self):
        # At HIGH risk band (60.0 score):
        # motorcycle clearance is lower -> higher risk
        # bus clearance is higher -> lower risk
        v_scores = compute_vehicle_risk_scores(60.0, "HIGH")
        assert v_scores["motorcycle"] > v_scores["car"]
        assert v_scores["car"] > v_scores["bus"]

    def test_score_network_end_to_end(self, sample_roads_gdf):
        scorer = RoadFloodRiskScorer()

        # Dynamic rainfall
        rain_df = pd.DataFrame({
            "rainfall_15min": [15.0],
            "rainfall_1h": [40.0],
            "rainfall_3h": [80.0],
            "rainfall_6h": [120.0],
            "rainfall_24h": [180.0],
            "rainfall_source_type": ["SIMULATED"],
        })

        # CCTV camera on road_1
        dw_df = pd.DataFrame({
            "segment_id": ["road_1"],
            "drainwatch_score": [0.9],
            "drainwatch_observation_class": ["drain_overflowing"],
            "drainwatch_confidence": [0.95],
            "drainwatch_data_age_hours": [0.5],
        })

        scored = scorer.score_network(
            roads_gdf=sample_roads_gdf,
            rainfall_df=rain_df,
            drainwatch_df=dw_df,
        )

        assert len(scored) == 3
        for col in [
            "flood_access_risk_score",
            "base_risk_score",
            "drainwatch_delta",
            "risk_class",
            "vehicle_scores",
            "prediction_set",
            "is_uncertain",
        ]:
            assert col in scored.columns

        # road_1 had drain_overflowing CCTV detection -> positive delta applied
        r1 = scored[scored["segment_id"] == "road_1"].iloc[0]
        assert r1["drainwatch_delta"] > 0
        assert r1["flood_access_risk_score"] >= r1["base_risk_score"]

        # road_2 had no camera -> delta must be 0
        r2 = scored[scored["segment_id"] == "road_2"].iloc[0]
        assert r2["drainwatch_delta"] == 0.0
        assert r2["flood_access_risk_score"] == r2["base_risk_score"]

    def test_scenario_extrapolation_detection(self, sample_roads_gdf):
        scorer = RoadFloodRiskScorer()
        # Extreme Michaung-scale scenario (430mm)
        extreme_rain_df = pd.DataFrame({
            "rainfall_15min": [25.0],
            "rainfall_1h": [80.0],
            "rainfall_3h": [150.0],
            "rainfall_6h": [250.0],
            "rainfall_24h": [430.0],
            "rainfall_source_type": ["SIMULATED"],
        })

        scored = scorer.score_network(sample_roads_gdf, rainfall_df=extreme_rain_df)
        assert all(scored["scenario_extrapolation"])
