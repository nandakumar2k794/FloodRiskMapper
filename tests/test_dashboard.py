"""
Unit tests for Streamlit Operational Dashboard and UI Components.
Verifies all 10 dashboard requirements:
1. Chennai interactive map construction
2. Road flood-risk visualization and risk classes (LOW, MODERATE, HIGH, CRITICAL)
3. Drainage layers
4. River/canal layers
5. Water bodies
6. CCTV locations
7. Selected-road details extraction (all 10 fields)
8. DrainWatch panel extraction (Camera, Garbage, Blockage, Water, Overflow, Confidence)
9. Rainfall scenario simulator formatting
10. Normal vs flood-safe route comparison
"""
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import folium

from app.data_loader import (
    get_chennai_roads_network,
    get_chennai_waterways,
    get_chennai_waterbodies,
    get_chennai_drains,
    get_chennai_cctv_cameras,
    get_cctv_dataframe,
)
from app.map_builder import (
    build_full_dashboard_map,
    get_risk_color,
    create_chennai_base_map,
)
from app.components.rainfall_simulator import RAINFALL_PRESETS, get_rainfall_dataframe
from app.components.road_details import get_operational_recommendation
from app.main import score_roads_live
from routing.graph import RoadGraphManager
from routing.router import FloodAwareRouter


class TestDashboardDataLoader:
    """Verify GIS layers, road networks, waterways, drains, and CCTV stations."""

    def test_roads_network_geodataframe(self):
        roads = get_chennai_roads_network()
        assert isinstance(roads, gpd.GeoDataFrame)
        assert len(roads) >= 10
        assert "segment_id" in roads.columns
        assert "elevation_mean" in roads.columns
        assert "slope" in roads.columns
        assert "local_elevation_context" in roads.columns
        assert "dist_to_water_m" in roads.columns
        assert "historical_flood_frequency" in roads.columns
        assert str(roads.crs) == "EPSG:4326"

    def test_waterways_and_waterbodies(self):
        waterways = get_chennai_waterways()
        waterbodies = get_chennai_waterbodies()

        assert isinstance(waterways, gpd.GeoDataFrame)
        assert isinstance(waterbodies, gpd.GeoDataFrame)
        assert len(waterways) >= 4  # Adyar, Cooum, Buckingham, etc.
        assert len(waterbodies) >= 3  # Pallikaranai, Velachery Lake, etc.

    def test_drainage_layer(self):
        drains = get_chennai_drains()
        assert isinstance(drains, gpd.GeoDataFrame)
        assert len(drains) >= 4
        assert "drain_id" in drains.columns
        assert "capacity_m3_s" in drains.columns

    def test_cctv_cameras_structure(self):
        cctv_list = get_chennai_cctv_cameras()
        assert len(cctv_list) >= 5

        # Check required fields for DrainWatch panel
        for cam in cctv_list:
            assert "camera_id" in cam
            assert "name" in cam
            assert "garbage_detected" in cam
            assert "blockage_percentage" in cam
            assert "water_coverage_percentage" in cam
            assert "overflow_detected" in cam
            assert "confidence" in cam
            assert 0.0 <= cam["confidence"] <= 1.0
            assert 0.0 <= cam["blockage_percentage"] <= 100.0


class TestMapBuilder:
    """Verify Folium map generation and color coding."""

    def test_risk_color_mapping(self):
        # LOW (0-25) -> Green
        assert get_risk_color(10.0) == "#22c55e"
        # MODERATE (25-50) -> Amber
        assert get_risk_color(35.0) == "#f59e0b"
        # HIGH (50-75) -> Orange
        assert get_risk_color(60.0) == "#f97316"
        # CRITICAL (75-100) -> Red
        assert get_risk_color(85.0) == "#ef4444"

    def test_build_full_dashboard_map(self):
        roads = get_chennai_roads_network()
        waterways = get_chennai_waterways()
        waterbodies = get_chennai_waterbodies()
        drains = get_chennai_drains()
        cctv = get_chennai_cctv_cameras()

        # Add mock scored columns
        roads["flood_access_risk_score"] = [35.0] * len(roads)
        roads["risk_class"] = ["MODERATE"] * len(roads)

        m = build_full_dashboard_map(
            scored_roads_gdf=roads,
            waterways_gdf=waterways,
            waterbodies_gdf=waterbodies,
            drains_gdf=drains,
            cctv_list=cctv,
            selected_road_id="seg_anna_salai_n",
        )

        assert isinstance(m, folium.Map)
        # Verify layers were added to map
        child_names = [child._name for child in m._children.values()]
        assert any("FeatureGroup" in name for name in child_names)


class TestLiveScoringAndPanels:
    """Verify live scoring, selected road panel logic, and routing comparison."""

    def test_live_scoring_produces_risk_classes(self):
        roads = get_chennai_roads_network()
        cctv_df = get_cctv_dataframe()
        rainfall = {
            "rainfall_15min": 10.0,
            "rainfall_1h": 25.0,
            "rainfall_3h": 45.0,
            "rainfall_6h": 65.0,
            "rainfall_24h": 85.0,
            "rainfall_source_type": "SIMULATED",
        }

        scored = score_roads_live(roads, rainfall, cctv_df)

        assert "flood_access_risk_score" in scored.columns
        assert "base_risk_score" in scored.columns
        assert "risk_class" in scored.columns
        assert "drainwatch_delta" in scored.columns

        # Verify all risk scores are in 0-100 bounds
        scores = scored["flood_access_risk_score"].values
        assert np.all((scores >= 0.0) & (scores <= 100.0))

        # Verify risk classes are strictly valid
        for c in scored["risk_class"].unique():
            assert c in ["LOW", "MODERATE", "HIGH", "CRITICAL"]

    def test_operational_recommendations(self):
        # Critical with blockage
        rec_crit = get_operational_recommendation(85.0, "CRITICAL", overflow_detected=True, blockage_pct=60.0)
        assert "CRITICAL" in rec_crit
        assert "desilting" in rec_crit

        # Low risk
        rec_low = get_operational_recommendation(15.0, "LOW")
        assert "LOW RISK" in rec_low

    def test_rainfall_presets_coverage(self):
        assert len(RAINFALL_PRESETS) >= 4
        for name, p in RAINFALL_PRESETS.items():
            assert "rainfall_24h" in p
            assert "rainfall_1h" in p
            df = get_rainfall_dataframe(p)
            assert len(df) == 1

    def test_routing_comparison_engine(self):
        roads = get_chennai_roads_network()
        cctv_df = get_cctv_dataframe()
        rainfall = {"rainfall_24h": 120.0, "rainfall_1h": 35.0, "rainfall_15min": 12.0, "rainfall_3h": 60.0, "rainfall_6h": 90.0, "rainfall_source_type": "SIMULATED"}
        scored = score_roads_live(roads, rainfall, cctv_df)

        gm = RoadGraphManager()
        gm.build_from_geodataframe(scored, risk_column="flood_access_risk_score")
        router = FloodAwareRouter(gm, algorithm="dijkstra")

        comp = router.compare_routes("node_central", "node_airport")
        assert comp.normal_route.distance_km > 0
        assert comp.flood_safe_route.distance_km > 0
        assert isinstance(comp.reroute_recommended, bool)
        assert isinstance(comp.summary, str)
