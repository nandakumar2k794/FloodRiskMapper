"""
Unit tests for ML Feature Matrix Assembly and Labeling Pipeline (Phase 4a).
Tests:
- Evidence tier sample weights assignment
- Scale pos weight calculation
- Real observed label generation
- Synthetic physics-based scenario labeling with noise injection
- Strict preservation of real observed labels
- Feature matrix construction across Tiers A-F
- Non-linear interaction terms calculation
- Structural handling of absent CCTV DrainWatch data
"""
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString

from ml.sample_weights import (
    calculate_scale_pos_weight,
    compute_sample_weights,
    WEIGHT_OBSERVED_POSITIVE,
    WEIGHT_INFERRED_POSITIVE,
    WEIGHT_SYNTHETIC_POSITIVE,
    WEIGHT_OBSERVED_NEGATIVE,
    WEIGHT_NO_EVIDENCE_NEGATIVE,
    WEIGHT_SYNTHETIC_NOISE_POSITIVE,
)
from ml.labeling import (
    generate_real_labels,
    generate_synthetic_scenario_labels,
    HIGHWAY_24H_RAIN_THRESHOLDS_MM,
)
from ml.feature_matrix import (
    FeatureMatrixBuilder,
    MODEL_FEATURE_NAMES,
    CATEGORICAL_FEATURES,
    DRAINWATCH_FEATURE_NAMES,
)


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
        },
        geometry=lines,
        crs="EPSG:4326",
    )


class TestSampleWeights:
    """Tests for evidence-tier sample weighting."""

    def test_scale_pos_weight(self):
        # 8 negatives, 2 positives -> ratio = 4.0
        y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        weight = calculate_scale_pos_weight(y)
        assert weight == pytest.approx(4.0)

    def test_scale_pos_weight_empty_or_no_positives(self):
        assert calculate_scale_pos_weight([]) == 1.0
        assert calculate_scale_pos_weight([0, 0, 0]) == 1.0

    def test_evidence_tier_weighting(self):
        df = pd.DataFrame({
            "road_inaccessible": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, np.nan],
            "primary_evidence_type": [
                "OBSERVED", "INFERRED", "SYNTHETIC", "SYNTHETIC",
                "OBSERVED", "NONE", "OBSERVED"
            ],
            "sample_source": ["REAL", "REAL", "SYNTHETIC", "SYNTHETIC", "REAL", "REAL", "REAL"],
            "is_synthetic_noise": [False, False, False, True, False, False, False],
        })
        weights = compute_sample_weights(df)
        
        # 1. Confirmed observed positive -> 1.0
        assert weights[0] == pytest.approx(WEIGHT_OBSERVED_POSITIVE)
        # 2. Inferred positive -> 0.7
        assert weights[1] == pytest.approx(WEIGHT_INFERRED_POSITIVE)
        # 3. Synthetic positive -> 0.5
        assert weights[2] == pytest.approx(WEIGHT_SYNTHETIC_POSITIVE)
        # 4. Synthetic noise positive -> 0.3
        assert weights[3] == pytest.approx(WEIGHT_SYNTHETIC_NOISE_POSITIVE)
        # 5. Confirmed negative -> 1.0
        assert weights[4] == pytest.approx(WEIGHT_OBSERVED_NEGATIVE)
        # 6. Absence of evidence negative -> 0.3
        assert weights[5] == pytest.approx(WEIGHT_NO_EVIDENCE_NEGATIVE)
        # 7. Unknown / NaN label -> 0.0 (masked)
        assert weights[6] == pytest.approx(0.0)


class TestLabeling:
    """Tests for real and synthetic labeling."""

    def test_generate_real_labels(self):
        df = pd.DataFrame({
            "historical_inundation_presence": [True, True, True, False],
            "primary_evidence_type": ["OBSERVED", "INFERRED", "OBSERVED", "NONE"],
            "historical_confidence": [0.90, 0.50, 0.40, 0.0],
        })
        labels = generate_real_labels(df, min_confidence=0.6)
        
        # Row 0: OBSERVED and conf=0.9 >= 0.6 -> 1.0
        assert labels.iloc[0] == 1.0
        # Row 1: INFERRED and conf=0.5 < 0.85 -> NaN
        assert pd.isna(labels.iloc[1])
        # Row 2: OBSERVED but conf=0.4 < 0.6 -> NaN
        assert pd.isna(labels.iloc[2])
        # Row 3: No flood -> NaN (absence does not imply dry)
        assert pd.isna(labels.iloc[3])

    def test_synthetic_scenario_labeling_physics(self):
        df = pd.DataFrame({
            "road_id": ["r_dep", "r_high", "r_water"],
            "highway": ["residential", "residential", "primary"],
            "rainfall_24h": [100.0, 100.0, 120.0],
            "local_elevation_context": [-2.5, 3.0, 0.0],  # r_dep is depression
            "dist_to_water_m": [500.0, 500.0, 30.0],      # r_water is near water
            "historical_flood_frequency": [0.0, 0.0, 0.0],
            "elevation_mean": [8.0, 25.0, 10.0],
            "slope": [0.5, 5.0, 1.0],
        })
        labeled = generate_synthetic_scenario_labels(df, noise_flip_rate=0.0)

        # r_dep has 100mm rain (>55mm for residential) AND local_context < -1.5 -> 1.0
        assert labeled.loc[labeled["road_id"] == "r_dep", "road_inaccessible"].iloc[0] == 1.0
        # r_high is at +3.0m elevation context, high slope -> 0.0
        assert labeled.loc[labeled["road_id"] == "r_high", "road_inaccessible"].iloc[0] == 0.0
        # r_water is near water (30m < 80m) with 120mm rain (>100mm) -> 1.0
        assert labeled.loc[labeled["road_id"] == "r_water", "road_inaccessible"].iloc[0] == 1.0

    def test_real_observed_precedence(self):
        """Verify real OBSERVED ground truth is NEVER overwritten by synthetic generation."""
        df = pd.DataFrame({
            "road_id": ["r_real_dry", "r_real_wet"],
            "highway": ["residential", "residential"],
            "rainfall_24h": [250.0, 20.0],
            "local_elevation_context": [-3.0, 5.0],
            "road_inaccessible": [0.0, 1.0],
            "sample_source": ["REAL", "REAL"],
        })
        labeled = generate_synthetic_scenario_labels(df, noise_flip_rate=0.0)

        # Even though r_real_dry meets synthetic flooding physics, its REAL label 0.0 is preserved
        row0 = labeled.loc[labeled["road_id"] == "r_real_dry"].iloc[0]
        assert row0["road_inaccessible"] == 0.0
        assert row0["sample_source"] == "REAL"

        # Even though r_real_wet has only 20mm rain, its REAL label 1.0 is preserved
        row1 = labeled.loc[labeled["road_id"] == "r_real_wet"].iloc[0]
        assert row1["road_inaccessible"] == 1.0
        assert row1["sample_source"] == "REAL"


class TestFeatureMatrixBuilder:
    """Tests for complete feature matrix assembly."""

    def test_assemble_and_combine(self, sample_roads_gdf):
        builder = FeatureMatrixBuilder()

        # Static terrain features
        terrain_df = pd.DataFrame({
            "segment_id": ["road_1", "road_2", "road_3"],
            "elevation_mean": [5.2, 12.1, 8.4],
            "elevation_min": [4.8, 11.5, 8.0],
            "elevation_max": [5.6, 12.8, 8.8],
            "elevation_range": [0.8, 1.3, 0.8],
            "slope": [0.4, 2.1, 1.2],
            "local_elevation_context": [-1.8, 1.5, -0.2],
        })

        # Static proximity features
        proximity_df = pd.DataFrame({
            "segment_id": ["road_1", "road_2", "road_3"],
            "dist_to_water_m": [45.0, 350.0, 120.0],
            "near_water_50m": [True, False, False],
            "near_water_200m": [True, False, True],
            "dist_to_drain_m": [12.0, 80.0, 25.0],
            "has_adjacent_drain": [True, False, False],
            "crosses_water": [False, False, False],
        })

        static_df = builder.assemble_static_road_features(
            roads_gdf=sample_roads_gdf,
            terrain_df=terrain_df,
            proximity_df=proximity_df,
        )
        assert len(static_df) == 3
        assert "elevation_mean" in static_df.columns
        assert "dist_to_water_m" in static_df.columns

        # Dynamic rainfall scenario
        rain_df = pd.DataFrame({
            "rainfall_15min": [10.0],
            "rainfall_1h": [35.0],
            "rainfall_3h": [75.0],
            "rainfall_6h": [110.0],
            "rainfall_24h": [180.0],
            "rainfall_source_type": ["SIMULATED"],
        })

        # Combine with dynamic rainfall
        combined = builder.combine_features(static_df, rainfall_df=rain_df)

        assert "rainfall_24h" in combined.columns
        assert combined["rainfall_24h"].iloc[0] == 180.0
        
        # Verify interaction terms
        assert "rainfall_24h_x_elevation_context" in combined.columns
        assert "rainfall_1h_x_dist_to_drain" in combined.columns
        expected_interaction = 180.0 * (-1.8)
        assert combined["rainfall_24h_x_elevation_context"].iloc[0] == pytest.approx(expected_interaction)

    def test_missing_drainwatch_handling(self, sample_roads_gdf):
        """Verify segments without cameras have NaN drainwatch_score and available=False."""
        builder = FeatureMatrixBuilder()
        static_df = builder.assemble_static_road_features(sample_roads_gdf)

        # Only road_1 has CCTV DrainWatch coverage
        dw_df = pd.DataFrame({
            "segment_id": ["road_1"],
            "drainwatch_score": [0.8],
            "drainwatch_confidence": [0.92],
            "drainwatch_data_age_hours": [0.5],
            "drainwatch_observation_class": ["drain_overflowing"],
        })

        combined = builder.combine_features(static_df, drainwatch_df=dw_df)

        # road_1 has camera
        r1 = combined[combined["road_id"] == "road_1"].iloc[0]
        assert r1["drainwatch_available"] is True or r1["drainwatch_available"] == 1
        assert r1["drainwatch_score"] == pytest.approx(0.8)

        # road_2 has no camera -> score must be NaN, available must be False
        r2 = combined[combined["road_id"] == "road_2"].iloc[0]
        assert (r2["drainwatch_available"] is False) or (r2["drainwatch_available"] == 0)
        assert np.isnan(r2["drainwatch_score"])

    def test_get_feature_matrix_numeric(self, sample_roads_gdf):
        """Verify extraction of clean numeric feature matrix X."""
        builder = FeatureMatrixBuilder()
        static_df = builder.assemble_static_road_features(sample_roads_gdf)
        combined = builder.combine_features(static_df)

        X, feature_cols = builder.get_feature_matrix(combined, encode_categoricals=True)

        assert isinstance(X, pd.DataFrame)
        assert len(X) == 3
        # Check core features are in X
        for f in ["rainfall_24h", "elevation_mean", "slope", "dist_to_water_m"]:
            assert f in X.columns
        # Check one-hot categorical encoding
        assert any(c.startswith("highway_") for c in X.columns)
        # Check no non-numeric types
        assert all(np.issubdtype(X[c].dtype, np.number) for c in X.columns)
