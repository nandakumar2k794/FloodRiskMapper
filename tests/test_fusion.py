"""
Unit tests for DrainWatch-to-Flood-Risk Fusion Architecture and Explanation Generation.
Verifies all requirements:
1. CCTV observations must be optional.
2. Missing CCTV must not break the model or explanation.
3. CCTV confidence must be considered and weighted.
4. Do not double-count features.
5. Preserve baseline GIS/rainfall flood risk.
6. Produce final road risk 0-100 and correct risk classes.
7. For every high-risk road generate structured 'WHY IS THIS ROAD HIGH RISK?' explanation:
   - Heavy rainfall
   - Low elevation
   - Historical vulnerability
   - Drain inlet blockage
   - Increasing water coverage
8. Structured explanation output serialization and formatting.
"""
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString

from ml.fusion import (
    DrainWatchFusionEngine,
    fuse_drainwatch_risk,
    get_fusion_engine,
)
from ml.explanation import (
    RiskExplanationGenerator,
    RoadRiskExplanation,
    RiskFactor,
)
from ml.drainwatch_adjustment import DrainWatchAdjuster, get_staleness_multiplier
from ml.risk_scorer import RoadFloodRiskScorer


class TestDrainWatchFusionEngine:
    """Core tests for DrainWatch-to-flood-risk fusion architecture."""

    def test_cctv_optional_none_preserves_base_score(self):
        """When drainwatch_features is None, baseline score is strictly preserved."""
        engine = DrainWatchFusionEngine()
        res = engine.fuse_single_road(
            base_risk_score=45.0,
            drainwatch_features=None,
            road_attributes={"elevation_mean": 8.0, "rainfall_24h": 50.0},
            road_id="road_101",
        )

        assert res["road_id"] == "road_101"
        assert res["base_risk_score"] == 45.0
        assert res["drainwatch_delta"] == 0.0
        assert res["flood_access_risk_score"] == 45.0
        assert res["cctv_applied"] is False
        assert res["cctv_confidence"] is None
        assert res["risk_class"] == "MODERATE"

    def test_cctv_unavailable_flag_preserves_base_score(self):
        """When drainwatch_available=False, delta is strictly 0.0."""
        engine = DrainWatchFusionEngine()
        dw = {
            "drainwatch_available": False,
            "drainwatch_score": 0.8,
            "drainwatch_observation_class": "road_waterlogged",
            "drainwatch_confidence": 0.95,
        }
        res = engine.fuse_single_road(
            base_risk_score=62.0,
            drainwatch_features=dw,
            road_id="road_102",
        )
        assert res["drainwatch_delta"] == 0.0
        assert res["flood_access_risk_score"] == 62.0
        assert res["cctv_applied"] is False

    def test_cctv_confidence_weighting(self):
        """Higher confidence produces proportionally higher delta."""
        engine = DrainWatchFusionEngine()
        dw_full_conf = {
            "drainwatch_available": True,
            "drainwatch_observation_class": "drain_blocked_garbage",
            "drainwatch_confidence": 1.0,
            "drainwatch_data_age_hours": 0.5,
        }
        res_full = engine.fuse_single_road(40.0, drainwatch_features=dw_full_conf)

        dw_half_conf = {
            "drainwatch_available": True,
            "drainwatch_observation_class": "drain_blocked_garbage",
            "drainwatch_confidence": 0.5,
            "drainwatch_data_age_hours": 0.5,
        }
        res_half = engine.fuse_single_road(40.0, drainwatch_features=dw_half_conf)

        # Full confidence should give +15.0 delta, half confidence should give +7.5 delta
        assert res_full["drainwatch_delta"] == pytest.approx(15.0)
        assert res_half["drainwatch_delta"] == pytest.approx(7.5)
        assert res_full["flood_access_risk_score"] > res_half["flood_access_risk_score"]

    def test_cctv_low_confidence_suppressed(self):
        """CCTV observations with confidence < 0.2 are suppressed as noise."""
        engine = DrainWatchFusionEngine()
        dw_noise = {
            "drainwatch_available": True,
            "drainwatch_observation_class": "drain_blocked_garbage",
            "drainwatch_confidence": 0.1,  # below noise floor
            "drainwatch_data_age_hours": 0.5,
        }
        res = engine.fuse_single_road(50.0, drainwatch_features=dw_noise)
        assert res["drainwatch_delta"] == 0.0
        assert res["flood_access_risk_score"] == 50.0

    def test_staleness_decay(self):
        """Observations decay over time and are zeroed out after 12 hours."""
        engine = DrainWatchFusionEngine()

        # 0-1h: 1.0x -> delta = 20.0
        res_fresh = engine.fuse_single_road(
            40.0,
            drainwatch_features={
                "drainwatch_available": True,
                "drainwatch_observation_class": "drain_overflowing",
                "drainwatch_confidence": 1.0,
                "drainwatch_data_age_hours": 0.5,
            },
        )
        assert res_fresh["drainwatch_delta"] == pytest.approx(20.0)

        # 1-4h: 0.7x -> delta = 14.0
        res_med = engine.fuse_single_road(
            40.0,
            drainwatch_features={
                "drainwatch_available": True,
                "drainwatch_observation_class": "drain_overflowing",
                "drainwatch_confidence": 1.0,
                "drainwatch_data_age_hours": 2.0,
            },
        )
        assert res_med["drainwatch_delta"] == pytest.approx(14.0)

        # 4-12h: 0.3x -> delta = 6.0
        res_old = engine.fuse_single_road(
            40.0,
            drainwatch_features={
                "drainwatch_available": True,
                "drainwatch_observation_class": "drain_overflowing",
                "drainwatch_confidence": 1.0,
                "drainwatch_data_age_hours": 6.0,
            },
        )
        assert res_old["drainwatch_delta"] == pytest.approx(6.0)

        # > 12h: 0.0x -> delta = 0.0
        res_expired = engine.fuse_single_road(
            40.0,
            drainwatch_features={
                "drainwatch_available": True,
                "drainwatch_observation_class": "drain_overflowing",
                "drainwatch_confidence": 1.0,
                "drainwatch_data_age_hours": 14.0,
            },
        )
        assert res_expired["drainwatch_delta"] == pytest.approx(0.0)
        assert res_expired["flood_access_risk_score"] == 40.0

    def test_continuous_indicators_rise_rate_and_blockage(self):
        """Continuous water rise rate and blockage % enhance the Stage 2 delta."""
        engine = DrainWatchFusionEngine()
        dw_rising = {
            "drainwatch_available": True,
            "drainwatch_observation_class": "drain_blocked_garbage",
            "drainwatch_confidence": 1.0,
            "drainwatch_data_age_hours": 0.2,
            "drainwatch_blockage_pct": 75.0,
            "drainwatch_water_rise_rate": 2.0,  # +2%/min water rise
        }
        res = engine.fuse_single_road(50.0, drainwatch_features=dw_rising)
        # Base class delta = 15.0 + velocity bonus min(6.0, 2.0 * 1.5) = 18.0
        assert res["drainwatch_delta"] == pytest.approx(18.0)
        assert res["flood_access_risk_score"] == 68.0

    def test_score_bounded_between_0_and_100(self):
        """Scores are strictly bounded in [0.0, 100.0]."""
        engine = DrainWatchFusionEngine()

        # Upper bound
        dw_high = {
            "drainwatch_available": True,
            "drainwatch_observation_class": "road_waterlogged",
            "drainwatch_confidence": 1.0,
            "drainwatch_data_age_hours": 0.1,
        }
        res_top = engine.fuse_single_road(92.0, drainwatch_features=dw_high)
        assert res_top["flood_access_risk_score"] == 100.0

        # Lower bound
        dw_low = {
            "drainwatch_available": True,
            "drainwatch_observation_class": "drain_clear",
            "drainwatch_confidence": 1.0,
            "drainwatch_data_age_hours": 0.1,
        }
        res_bottom = engine.fuse_single_road(3.0, drainwatch_features=dw_low)
        assert res_bottom["flood_access_risk_score"] == 0.0

    def test_risk_classes_mapping(self):
        """Verify all standard risk classes: LOW, MODERATE, HIGH, CRITICAL."""
        engine = DrainWatchFusionEngine()
        assert engine.fuse_single_road(15.0)["risk_class"] == "LOW"
        assert engine.fuse_single_road(35.0)["risk_class"] == "MODERATE"
        assert engine.fuse_single_road(60.0)["risk_class"] == "HIGH"
        assert engine.fuse_single_road(85.0)["risk_class"] == "CRITICAL"


class TestExplanationGeneration:
    """Tests for structured 'WHY IS THIS ROAD HIGH RISK?' explanations."""

    def test_high_risk_road_headline_and_exact_factors(self):
        """
        Verify that a high-risk road generates the exact required headline:
        'WHY IS THIS ROAD HIGH RISK?'
        and triggers all 5 required factor examples when conditions are present:
        - Heavy rainfall
        - Low elevation
        - Historical vulnerability
        - Drain inlet blockage
        - Increasing water coverage
        """
        generator = RiskExplanationGenerator()
        road_data = {
            "road_id": "nh4_koyambedu",
            "base_risk_score": 65.0,
            "flood_access_risk_score": 82.0,
            "risk_class": "CRITICAL",
            # 1. Heavy rainfall
            "rainfall_24h": 140.0,
            "rainfall_1h": 22.0,
            "rainfall_source_type": "REAL",
            # 2. Low elevation
            "elevation_mean": 3.2,
            "local_elevation_context": -1.8,
            "slope": 0.4,
            "hand_value": 1.5,
            # 3. Historical vulnerability
            "historical_flood_count": 3,
            "historical_flood_frequency": 0.65,
            "primary_evidence_type": "OBSERVED",
            # 4. Drain inlet blockage & 5. Increasing water coverage (CCTV)
            "drainwatch_available": True,
            "drainwatch_confidence": 0.92,
            "drainwatch_data_age_hours": 0.5,
            "drainwatch_observation_class": "drain_blocked_garbage",
            "drainwatch_blockage_pct": 70.0,
            "drainwatch_water_coverage_pct": 45.0,
            "drainwatch_water_rise_rate": 2.4,
            "drainwatch_overflow_detected": False,
        }

        explanation = generator.explain_road(road_data)

        # 1. Verify headline
        assert explanation.headline == "WHY IS THIS ROAD HIGH RISK?"
        assert explanation.is_high_risk is True
        assert explanation.risk_class == "CRITICAL"
        assert explanation.risk_score == 82.0

        # 2. Verify exact factor names
        expected_factors = [
            "Heavy rainfall",
            "Low elevation",
            "Historical vulnerability",
            "Drain inlet blockage",
            "Increasing water coverage",
        ]
        for factor in expected_factors:
            assert factor in explanation.factors, f"Missing factor: {factor}"

        # 3. Verify structured factors detail
        factors_by_name = {f.factor_name: f for f in explanation.structured_factors}

        # Check Heavy rainfall factor
        f_rain = factors_by_name["Heavy rainfall"]
        assert f_rain.category == "RAINFALL"
        assert "140.0 mm" in f_rain.contributing_value
        assert f_rain.source_type == "REAL"
        assert f_rain.severity in ["HIGH", "CRITICAL"]

        # Check Low elevation factor
        f_elev = factors_by_name["Low elevation"]
        assert f_elev.category == "TERRAIN"
        assert "depression" in f_elev.contributing_value
        assert f_elev.severity in ["HIGH", "CRITICAL"]

        # Check Historical vulnerability factor
        f_hist = factors_by_name["Historical vulnerability"]
        assert f_hist.category == "HISTORICAL"
        assert "3 prior inundation" in f_hist.contributing_value
        assert f_hist.source_type == "OBSERVED"

        # Check Drain inlet blockage factor
        f_block = factors_by_name["Drain inlet blockage"]
        assert f_block.category == "DRAINWATCH_CV"
        assert "70.0% blockage" in f_block.contributing_value
        assert f_block.source_type == "OBSERVED_CV"

        # Check Increasing water coverage factor
        f_rise = factors_by_name["Increasing water coverage"]
        assert f_rise.category == "DRAINWATCH_CV"
        assert "+2.40%/min" in f_rise.contributing_value
        assert f_rise.source_type == "OBSERVED_CV"

        # 4. Verify text formatting
        text_output = explanation.format_text()
        assert "=== WHY IS THIS ROAD HIGH RISK? ===" in text_output
        assert "Heavy rainfall" in text_output
        assert "Low elevation" in text_output
        assert "Drain inlet blockage" in text_output
        assert "Increasing water coverage" in text_output

    def test_missing_cctv_explanation_preserves_baseline_factors(self):
        """
        When CCTV is not available, explanation generates cleanly from GIS/rainfall/historical
        data without crashing or hallucinating CCTV factors.
        """
        generator = RiskExplanationGenerator()
        road_data = {
            "road_id": "velachery_main_rd",
            "base_risk_score": 72.0,
            "flood_access_risk_score": 72.0,
            "risk_class": "HIGH",
            "rainfall_24h": 95.0,
            "rainfall_1h": 18.0,
            "elevation_mean": 2.5,
            "local_elevation_context": -2.0,
            "historical_flood_count": 2,
            "historical_flood_frequency": 0.40,
            # No CCTV
            "drainwatch_available": False,
        }

        explanation = generator.explain_road(road_data)

        assert explanation.headline == "WHY IS THIS ROAD HIGH RISK?"
        assert explanation.cctv_available is False
        assert explanation.cctv_confidence is None

        # Baseline factors present
        assert "Heavy rainfall" in explanation.factors
        assert "Low elevation" in explanation.factors
        assert "Historical vulnerability" in explanation.factors

        # CCTV factors must NOT be present
        assert "Drain inlet blockage" not in explanation.factors
        assert "Increasing water coverage" not in explanation.factors
        assert "DRAINWATCH_CCTV" not in explanation.data_sources

    def test_no_double_counting_features(self):
        """
        Ensure distinct feature sources: rainfall, terrain, historical, and CCTV
        do not overlap or duplicate categories in the structured output.
        """
        generator = RiskExplanationGenerator()
        road_data = {
            "road_id": "road_distinct",
            "flood_access_risk_score": 78.0,
            "rainfall_24h": 85.0,
            "elevation_mean": 3.0,
            "historical_flood_count": 1,
            "drainwatch_available": True,
            "drainwatch_confidence": 0.85,
            "drainwatch_blockage_pct": 50.0,
            "drainwatch_water_rise_rate": 1.2,
        }
        explanation = generator.explain_road(road_data)

        # Check each factor appears exactly once
        assert len(explanation.factors) == len(set(explanation.factors))

        categories = [f.category for f in explanation.structured_factors]
        # RAINFALL, TERRAIN, HISTORICAL are separate categories
        assert categories.count("RAINFALL") <= 1
        assert categories.count("TERRAIN") <= 1
        assert categories.count("HISTORICAL") <= 1

    def test_dict_serialization_complete(self):
        """Verify to_dict() contains all required structured fields."""
        engine = DrainWatchFusionEngine()
        res = engine.fuse_single_road(
            base_risk_score=75.0,
            drainwatch_features={
                "drainwatch_available": True,
                "drainwatch_observation_class": "drain_blocked_garbage",
                "drainwatch_confidence": 0.90,
                "drainwatch_data_age_hours": 0.3,
                "drainwatch_blockage_pct": 55.0,
                "drainwatch_water_rise_rate": 1.8,
            },
            road_attributes={
                "rainfall_24h": 120.0,
                "elevation_mean": 3.0,
                "local_elevation_context": -1.2,
                "historical_flood_count": 2,
            },
            road_id="rd_test_serial",
        )

        d = res["explanation"]
        assert isinstance(d, dict)
        assert d["road_id"] == "rd_test_serial"
        assert d["headline"] == "WHY IS THIS ROAD HIGH RISK?"
        assert isinstance(d["risk_score"], float)
        assert isinstance(d["risk_class"], str)
        assert isinstance(d["is_high_risk"], bool)
        assert isinstance(d["factors"], list)
        assert isinstance(d["structured_factors"], list)
        assert isinstance(d["summary_text"], str)
        assert isinstance(d["data_sources"], list)
        assert d["cctv_available"] is True
        assert d["cctv_confidence"] == pytest.approx(0.90)


class TestBatchAndNetworkFusion:
    """Tests for DataFrame and NetworkX/GeoPandas integration."""

    def test_fuse_dataframe(self):
        """Batch fusion correctly handles mix of CCTV and non-CCTV roads."""
        engine = DrainWatchFusionEngine()
        df = pd.DataFrame({
            "road_id": ["r1", "r2", "r3"],
            "base_risk_score": [60.0, 30.0, 70.0],
            "rainfall_24h": [100.0, 10.0, 150.0],
            "elevation_mean": [3.0, 15.0, 2.0],
            "local_elevation_context": [-1.0, 2.0, -2.5],
            "historical_flood_count": [2, 0, 4],
            "drainwatch_available": [True, False, True],
            "drainwatch_observation_class": ["drain_blocked_garbage", None, "drain_overflowing"],
            "drainwatch_confidence": [0.9, np.nan, 0.95],
            "drainwatch_data_age_hours": [0.5, np.nan, 0.2],
            "drainwatch_blockage_pct": [65.0, 0.0, 20.0],
            "drainwatch_water_rise_rate": [1.5, 0.0, 3.0],
        })

        fused = engine.fuse_dataframe(df)

        assert "flood_access_risk_score" in fused.columns
        assert "drainwatch_delta" in fused.columns
        assert "risk_class" in fused.columns
        assert "explanation" in fused.columns
        assert "risk_factors" in fused.columns

        # r1: CCTV camera blocked -> delta > 0 -> adjusted > base
        r1 = fused.iloc[0]
        assert r1["drainwatch_delta"] > 0
        assert r1["flood_access_risk_score"] > r1["base_risk_score"]
        assert "Drain inlet blockage" in r1["risk_factors"]

        # r2: No CCTV -> delta == 0 -> base score preserved
        r2 = fused.iloc[1]
        assert r2["drainwatch_delta"] == 0.0
        assert r2["flood_access_risk_score"] == 30.0
        assert "Drain inlet blockage" not in r2["risk_factors"]

        # r3: High rainfall + low elevation + overflow -> CRITICAL
        r3 = fused.iloc[2]
        assert r3["risk_class"] == "CRITICAL"
        assert r3["explanation"]["headline"] == "WHY IS THIS ROAD HIGH RISK?"

    def test_end_to_end_risk_scorer_includes_explanations(self):
        """RoadFloodRiskScorer.score_network enriches output with explanation and risk_factors."""
        lines = [
            LineString([(80.20, 13.00), (80.21, 13.00)]),
            LineString([(80.22, 13.02), (80.23, 13.02)]),
        ]
        roads_gdf = gpd.GeoDataFrame(
            {
                "segment_id": ["road_alpha", "road_beta"],
                "highway": ["primary", "residential"],
                "elevation_mean": [2.5, 12.0],
                "slope": [0.5, 2.0],
                "local_elevation_context": [-2.0, 1.0],
                "historical_flood_frequency": [0.5, 0.0],
            },
            geometry=lines,
            crs="EPSG:4326",
        )

        rain_df = pd.DataFrame({
            "rainfall_15min": [10.0],
            "rainfall_1h": [30.0],
            "rainfall_3h": [60.0],
            "rainfall_6h": [90.0],
            "rainfall_24h": [120.0],
            "rainfall_source_type": ["SIMULATED"],
        })

        dw_df = pd.DataFrame({
            "segment_id": ["road_alpha"],
            "drainwatch_score": [0.9],
            "drainwatch_observation_class": ["drain_blocked_garbage"],
            "drainwatch_confidence": [0.95],
            "drainwatch_data_age_hours": [0.2],
            "drainwatch_blockage_pct": [60.0],
            "drainwatch_water_rise_rate": [1.5],
        })

        scorer = RoadFloodRiskScorer()
        scored = scorer.score_network(roads_gdf, rainfall_df=rain_df, drainwatch_df=dw_df)

        assert "explanation" in scored.columns
        assert "risk_factors" in scored.columns

        # road_alpha should have explanation with headline and factors
        r_alpha = scored[scored["segment_id"] == "road_alpha"].iloc[0]
        exp_alpha = r_alpha["explanation"]
        assert exp_alpha["road_id"] == "road_alpha"
        assert len(r_alpha["risk_factors"]) > 0

    def test_convenience_fuse_function(self):
        """Test fuse_drainwatch_risk standalone function."""
        res = fuse_drainwatch_risk(
            base_risk_score=70.0,
            drainwatch_features={
                "drainwatch_available": True,
                "drainwatch_observation_class": "road_waterlogged",
                "drainwatch_confidence": 0.9,
                "drainwatch_data_age_hours": 0.5,
            },
            road_id="rd_standalone",
        )
        assert res["road_id"] == "rd_standalone"
        assert res["flood_access_risk_score"] > 70.0
        assert res["is_high_risk"] is True
        assert res["explanation"]["headline"] == "WHY IS THIS ROAD HIGH RISK?"
