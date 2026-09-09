"""
Unit and integration tests for the Historical Flood Pipeline.
Verifies:
- Separation of OBSERVED, INFERRED, and SYNTHETIC evidence tiers.
- Calculation of historical_flood_count, historical_flood_frequency, historical_inundation_presence.
- Spatial uncertainty and proximity buffering.
- Handling of absence (absence is NOT treated as proof of zero flood risk).
"""
import pytest
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Polygon, Point

from config.constants import CRS_WGS84
from historical.models import (
    HistoricalEvidenceType,
    HistoricalFloodEvent,
)
from historical.ingestion import (
    HistoricalFloodIngestor,
    CHENNAI_HISTORICAL_EVENTS,
)
from historical.association import associate_historical_floods_with_roads
from historical.features import (
    HistoricalFloodFeaturePipeline,
    HISTORICAL_FEATURE_COLUMNS,
)


# ─── 1. Evidence Tier Separation Tests ───────────────────────────────────────

class TestEvidenceSeparation:
    """Verify distinct handling of OBSERVED, INFERRED, and SYNTHETIC tiers."""

    def test_evidence_types_enum(self):
        assert HistoricalEvidenceType.OBSERVED.value == "OBSERVED"
        assert HistoricalEvidenceType.INFERRED.value == "INFERRED"
        assert HistoricalEvidenceType.SYNTHETIC.value == "SYNTHETIC"

    def test_registered_historical_events(self):
        ingestor = HistoricalFloodIngestor()
        events = ingestor.get_registered_events()
        assert len(events) >= 3

        event_ids = [e.event_id for e in events]
        assert "EVENT_2015_DEC" in event_ids
        assert "EVENT_2021_NOV" in event_ids
        assert "EVENT_2023_MICHAUNG" in event_ids


# ─── 2. Ingestion & File Handling Tests ──────────────────────────────────────

class TestHistoricalIngestion:
    """Verify ingestion file handling without data fabrication."""

    def test_load_non_existent_file_raises_not_found(self):
        ingestor = HistoricalFloodIngestor()
        with pytest.raises(FileNotFoundError, match="Historical flood file not found"):
            ingestor.load_flood_dataset("missing_flood.geojson", event_id="EVENT_2015_DEC")

    def test_load_valid_file(self, tmp_path):
        poly = Polygon([(80.24, 13.04), (80.26, 13.04), (80.26, 13.06), (80.24, 13.06), (80.24, 13.04)])
        gdf = gpd.GeoDataFrame([{"name": "Velachery 2015 Flooded Zone", "geometry": poly}], crs=CRS_WGS84)
        file_path = tmp_path / "hist_flood.geojson"
        file_path.write_text(gdf.to_json(), encoding="utf-8")

        ingestor = HistoricalFloodIngestor()
        loaded = ingestor.load_flood_dataset(
            file_path,
            event_id="EVENT_2015_DEC",
            evidence_type=HistoricalEvidenceType.OBSERVED,
        )
        assert len(loaded) == 1
        assert loaded.iloc[0]["event_id"] == "EVENT_2015_DEC"
        assert loaded.iloc[0]["evidence_type"] == "OBSERVED"


# ─── 3. Spatial Association & Feature Calculation Tests ──────────────────────

class TestHistoricalFeatureCalculation:
    """Verify flood count, frequency, presence, and uncertainty handling."""

    def test_direct_observed_inundation(self):
        # Road 1 inside 2015 and 2023 flood polygons
        # Road 2 outside both polygons
        road_flooded = LineString([(80.245, 13.045), (80.255, 13.055)])
        road_dry = LineString([(80.210, 13.100), (80.220, 13.100)])

        roads_gdf = gpd.GeoDataFrame(
            [
                {"road_id": "r_velachery", "geometry": road_flooded},
                {"road_id": "r_dry_ridge", "geometry": road_dry},
            ],
            crs=CRS_WGS84,
        )

        # 2 distinct flood polygons (2015 and 2023)
        poly_2015 = Polygon([(80.240, 13.040), (80.260, 13.040), (80.260, 13.060), (80.240, 13.060)])
        poly_2023 = Polygon([(80.242, 13.042), (80.258, 13.042), (80.258, 13.058), (80.242, 13.058)])

        flood_gdf = gpd.GeoDataFrame(
            [
                {"event_id": "EVENT_2015_DEC", "evidence_type": "OBSERVED", "geometry": poly_2015},
                {"event_id": "EVENT_2023_MICHAUNG", "evidence_type": "OBSERVED", "geometry": poly_2023},
            ],
            crs=CRS_WGS84,
        )

        pipeline = HistoricalFloodFeaturePipeline(total_benchmark_events=3)
        result = pipeline.extract_features(roads_gdf, flood_gdf)

        # Check required columns
        for col in HISTORICAL_FEATURE_COLUMNS:
            assert col in result.columns

        r_fl = result[result["road_id"] == "r_velachery"].iloc[0]
        r_dr = result[result["road_id"] == "r_dry_ridge"].iloc[0]

        # Flooded road metrics
        assert r_fl["historical_flood_count"] == 2
        assert r_fl["observed_flood_count"] == 2
        assert r_fl["historical_flood_frequency"] == pytest.approx(2 / 3, rel=1e-2)
        assert bool(r_fl["historical_inundation_presence"]) is True
        assert r_fl["primary_evidence_type"] == "OBSERVED"
        assert r_fl["historical_confidence"] == 1.0

        # Dry road metrics (Absence is recorded, confidence of safety is low rather than proven immunity)
        assert r_dr["historical_flood_count"] == 0
        assert r_dr["observed_flood_count"] == 0
        assert r_dr["historical_flood_frequency"] == 0.0
        assert bool(r_dr["historical_inundation_presence"]) is False
        assert r_dr["min_dist_to_flood_record_m"] > 1000.0
        assert r_dr["historical_confidence"] == 0.0  # reflects lack of direct observation

    def test_inferred_proximity_flooding(self):
        # Road near flood polygon (within 50m buffer, but not directly intersecting)
        flood_poly = Polygon([(80.250, 13.050), (80.252, 13.050), (80.252, 13.052), (80.250, 13.052)])
        flood_gdf = gpd.GeoDataFrame(
            [{"event_id": "EVENT_2015_DEC", "evidence_type": "OBSERVED", "geometry": flood_poly}],
            crs=CRS_WGS84,
        )

        # Road ~30m south of polygon
        near_road = LineString([(80.2501, 13.0496), (80.2519, 13.0496)])
        roads_gdf = gpd.GeoDataFrame([{"road_id": "r_near", "geometry": near_road}], crs=CRS_WGS84)

        pipeline = HistoricalFloodFeaturePipeline(direct_tolerance_m=10.0, uncertainty_buffer_m=80.0)
        result = pipeline.extract_features(roads_gdf, flood_gdf)
        row = result.iloc[0]

        # Should be classified as INFERRED due to proximity buffer
        assert bool(row["historical_inundation_presence"]) is True
        assert row["inferred_flood_count"] >= 1
        assert row["primary_evidence_type"] == "INFERRED"
        assert 0.4 <= row["historical_confidence"] <= 0.9

    def test_synthetic_tier_separation(self):
        # Road inside a 100-year synthetic hydrodynamic simulation zone
        syn_poly = Polygon([(80.240, 13.040), (80.260, 13.040), (80.260, 13.060), (80.240, 13.060)])
        flood_gdf = gpd.GeoDataFrame(
            [{"event_id": "SYN_100YR_FLOOD", "evidence_type": "SYNTHETIC", "geometry": syn_poly}],
            crs=CRS_WGS84,
        )

        road = LineString([(80.245, 13.045), (80.255, 13.055)])
        roads_gdf = gpd.GeoDataFrame([{"road_id": "r_syn", "geometry": road}], crs=CRS_WGS84)

        pipeline = HistoricalFloodFeaturePipeline(total_benchmark_events=1)
        result = pipeline.extract_features(roads_gdf, flood_gdf)
        row = result.iloc[0]

        assert row["observed_flood_count"] == 0
        assert row["inferred_flood_count"] == 0
        assert row["synthetic_flood_count"] == 1
        assert row["primary_evidence_type"] == "SYNTHETIC"
        assert row["historical_confidence"] == 0.65

    def test_empty_roads_returns_empty_schema(self):
        empty_roads = gpd.GeoDataFrame(columns=["road_id", "geometry"], crs=CRS_WGS84)
        pipeline = HistoricalFloodFeaturePipeline()
        result = pipeline.extract_features(empty_roads)
        assert len(result) == 0
        for col in HISTORICAL_FEATURE_COLUMNS:
            assert col in result.columns
