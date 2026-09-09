"""
Unit and integration tests for GIS Foundation.
Tests CRS handling, geometry validation, road loading, spatial joins,
proximity analysis, and separate processed storage.
"""
import pytest
import tempfile
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString, Polygon

from config.constants import CHENNAI_BBOX, CRS_WGS84, CRS_UTM44N
from gis.preprocessing.crs import ensure_crs, to_metric, to_geographic, is_metric_crs
from gis.preprocessing.geometry import (
    validate_and_repair_geometries,
    filter_by_bbox,
    compute_metric_dimensions,
)
from gis.preprocessing.cleaners import clean_roads, clean_waterways, clean_boundaries
from gis.features.spatial_joins import assign_zone_and_ward, detect_water_crossings
from gis.features.proximity import calculate_distance_to_water, calculate_distance_to_drains
from gis.ingestion.roads import FileRoadAdapter, OSMRoadAdapter
from gis.ingestion.water import FileWaterAdapter, OSMWaterAdapter
from gis.ingestion.drainage import FileDrainageAdapter, EXPECTED_SWD_COLUMNS
from gis.ingestion.boundary import FileBoundaryAdapter, EXPECTED_BOUNDARY_COLUMNS
from gis.ingestion.elevation import ElevationAdapter
from gis.storage import save_processed_layer, load_processed_layer


# ─── 1. CRS Handling Tests ───────────────────────────────────────────────────

class TestCRSHandling:
    """Verify reprojection and CRS management."""

    def test_ensure_crs_assigns_wgs84(self):
        # Unprojected GeoDataFrame
        geom = [Point(80.2707, 13.0827)]
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=geom, crs=None)

        result = ensure_crs(gdf, target_crs=CRS_WGS84)
        assert result.crs is not None
        assert result.crs.to_string() == CRS_WGS84

    def test_reprojection_wgs84_to_utm44n(self):
        # Chennai Central coordinates
        geom = [Point(80.2707, 13.0827)]
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=geom, crs=CRS_WGS84)

        gdf_utm = to_metric(gdf)
        assert gdf_utm.crs.to_string() == CRS_UTM44N
        assert is_metric_crs(gdf_utm) is True

        # UTM Zone 44N Easting should be roughly 420,000m and Northing ~1,446,000m
        pt = gdf_utm.geometry.iloc[0]
        assert 400_000 < pt.x < 450_000
        assert 1_400_000 < pt.y < 1_500_000

    def test_roundtrip_transformation(self):
        geom = [LineString([(80.25, 13.05), (80.26, 13.06)])]
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=geom, crs=CRS_WGS84)

        gdf_metric = to_metric(gdf)
        gdf_back = to_geographic(gdf_metric)

        orig_coords = list(gdf.geometry.iloc[0].coords)
        back_coords = list(gdf_back.geometry.iloc[0].coords)
        for (x1, y1), (x2, y2) in zip(orig_coords, back_coords):
            assert abs(x1 - x2) < 1e-6
            assert abs(y1 - y2) < 1e-6


# ─── 2. Geometry Validity Tests ──────────────────────────────────────────────

class TestGeometryValidity:
    """Verify geometry repair, empty dropping, and bounding box filtering."""

    def test_repair_invalid_bowtie_polygon(self):
        # Self-intersecting polygon (bowtie) is topologically invalid
        bowtie = Polygon([(0, 0), (0, 2), (2, 0), (2, 2), (0, 0)])
        assert not bowtie.is_valid

        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[bowtie], crs=CRS_WGS84)
        repaired = validate_and_repair_geometries(gdf)

        assert len(repaired) == 1
        assert repaired.geometry.iloc[0].is_valid

    def test_drops_empty_and_null_geometries(self):
        geoms = [
            Point(80.25, 13.05),
            Polygon(),  # empty
            None,       # null
            LineString([(80.2, 13.0), (80.21, 13.01)]),
        ]
        gdf = gpd.GeoDataFrame({"id": [1, 2, 3, 4]}, geometry=geoms, crs=CRS_WGS84)
        cleaned = validate_and_repair_geometries(gdf)

        assert len(cleaned) == 2
        assert list(cleaned["id"]) == [1, 4]

    def test_filter_by_chennai_bbox(self):
        chennai_pt = Point(80.25, 13.05)      # inside Chennai
        delhi_pt = Point(77.20, 28.61)        # outside Chennai

        gdf = gpd.GeoDataFrame({"city": ["Chennai", "Delhi"]}, geometry=[chennai_pt, delhi_pt], crs=CRS_WGS84)
        filtered = filter_by_bbox(gdf, bbox=CHENNAI_BBOX)

        assert len(filtered) == 1
        assert filtered.iloc[0]["city"] == "Chennai"

    def test_metric_dimensions_calculation(self):
        # 1 km line in UTM 44N
        p1 = Point(420_000, 1_446_000)
        p2 = Point(421_000, 1_446_000)
        line = LineString([p1, p2])

        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[line], crs=CRS_UTM44N)
        gdf = compute_metric_dimensions(gdf)

        assert abs(gdf.iloc[0]["length_m"] - 1000.0) < 1.0


# ─── 3. Road Loading & Cleaning Tests ────────────────────────────────────────

class TestRoadLoading:
    """Verify road adapter loading, imputation, and normalization."""

    def test_file_road_adapter_missing_file_raises(self):
        adapter = FileRoadAdapter("non_existent_roads.geojson")
        assert not adapter.validate_source()
        with pytest.raises(FileNotFoundError):
            adapter.load()

    def test_clean_roads_imputation(self):
        # Simulated raw road segments with missing lanes and speeds
        roads_data = [
            {
                "segment_id": "seg_1",
                "highway": "primary",
                "name": "Anna Salai",
                "lanes": None,        # missing
                "maxspeed": None,     # missing
                "geometry": LineString([(80.25, 13.05), (80.26, 13.06)]),
            },
            {
                "segment_id": "seg_2",
                "highway": "residential",
                "name": "Gandhi Street",
                "lanes": "1",
                "maxspeed": "30 km/h",
                "geometry": LineString([(80.26, 13.06), (80.27, 13.07)]),
            },
        ]
        raw_gdf = gpd.GeoDataFrame(roads_data, crs=CRS_WGS84)
        cleaned = clean_roads(raw_gdf)

        # Anna Salai (primary) should be imputed with 2 lanes and 50 km/h
        row1 = cleaned[cleaned["segment_id"] == "seg_1"].iloc[0]
        assert row1["lanes"] == 2.0
        assert row1["maxspeed"] == 50.0
        assert row1["length_m"] > 0

        # Gandhi Street should parse 30 km/h into float 30.0
        row2 = cleaned[cleaned["segment_id"] == "seg_2"].iloc[0]
        assert row2["maxspeed"] == 30.0

    def test_osm_road_adapter_bbox_validation(self):
        adapter = OSMRoadAdapter(bbox=CHENNAI_BBOX)
        assert adapter.validate_source() is True

        bad_adapter = OSMRoadAdapter(bbox={"north": 10.0, "south": 12.0, "east": 80.0, "west": 81.0})
        assert bad_adapter.validate_source() is False


# ─── 4. Spatial Joins & Proximity Tests ──────────────────────────────────────

class TestSpatialJoins:
    """Verify spatial joins between roads, administrative zones, and water features."""

    def test_assign_zone_and_ward(self):
        # Create 2 zones in Chennai
        # Zone 9: Teynampet
        poly_z9 = Polygon([(80.23, 13.03), (80.26, 13.03), (80.26, 13.06), (80.23, 13.06), (80.23, 13.03)])
        # Zone 13: Adyar
        poly_z13 = Polygon([(80.23, 13.00), (80.26, 13.00), (80.26, 13.03), (80.23, 13.03), (80.23, 13.00)])

        zones_gdf = gpd.GeoDataFrame(
            [
                {"zone_no": 9, "zone_name": "Teynampet", "geometry": poly_z9},
                {"zone_no": 13, "zone_name": "Adyar", "geometry": poly_z13},
            ],
            crs=CRS_WGS84,
        )

        # Road 1 in Teynampet, Road 2 in Adyar
        road1 = LineString([(80.24, 13.04), (80.25, 13.05)])
        road2 = LineString([(80.24, 13.01), (80.25, 13.02)])

        roads_gdf = gpd.GeoDataFrame(
            [
                {"segment_id": "r1", "geometry": road1},
                {"segment_id": "r2", "geometry": road2},
            ],
            crs=CRS_WGS84,
        )

        joined = assign_zone_and_ward(roads_gdf, zones_gdf)

        r1_match = joined[joined["segment_id"] == "r1"].iloc[0]
        r2_match = joined[joined["segment_id"] == "r2"].iloc[0]

        assert r1_match["zone_no"] == 9
        assert r1_match["zone_name"] == "Teynampet"
        assert r2_match["zone_no"] == 13
        assert r2_match["zone_name"] == "Adyar"

    def test_detect_water_crossings(self):
        # Adyar River flowing east-west
        river = LineString([(80.20, 13.01), (80.28, 13.01)])
        water_gdf = gpd.GeoDataFrame(
            [{"name": "Adyar River", "water_category": "RIVER", "geometry": river}],
            crs=CRS_WGS84,
        )

        # Bridge crossing river north-south
        bridge_road = LineString([(80.24, 13.00), (80.24, 13.02)])
        # Parallel road that does not cross
        parallel_road = LineString([(80.24, 13.03), (80.25, 13.03)])

        roads_gdf = gpd.GeoDataFrame(
            [
                {"segment_id": "bridge", "geometry": bridge_road},
                {"segment_id": "parallel", "geometry": parallel_road},
            ],
            crs=CRS_WGS84,
        )

        result = detect_water_crossings(roads_gdf, water_gdf)

        bridge_row = result[result["segment_id"] == "bridge"].iloc[0]
        parallel_row = result[result["segment_id"] == "parallel"].iloc[0]

        assert bool(bridge_row["crosses_water"]) is True
        assert bridge_row["intersecting_water_type"] == "RIVER"
        assert bool(parallel_row["crosses_water"]) is False

    def test_calculate_distance_to_water(self):
        # Water body at specific coordinates
        water_poly = Polygon([(80.250, 13.050), (80.252, 13.050), (80.252, 13.052), (80.250, 13.052), (80.250, 13.050)])
        water_gdf = gpd.GeoDataFrame([{"name": "Lake", "geometry": water_poly}], crs=CRS_WGS84)

        # Road very close (~20m)
        close_road = LineString([(80.2501, 13.0498), (80.2519, 13.0498)])
        # Road far away (~1.5 km)
        far_road = LineString([(80.2500, 13.0650), (80.2520, 13.0650)])

        roads_gdf = gpd.GeoDataFrame(
            [
                {"segment_id": "close", "geometry": close_road},
                {"segment_id": "far", "geometry": far_road},
            ],
            crs=CRS_WGS84,
        )

        result = calculate_distance_to_water(roads_gdf, water_gdf)
        close_res = result[result["segment_id"] == "close"].iloc[0]
        far_res = result[result["segment_id"] == "far"].iloc[0]

        assert close_res["dist_to_water_m"] < 50.0
        assert bool(close_res["near_water_50m"]) is True
        assert far_res["dist_to_water_m"] > 500.0
        assert bool(far_res["near_water_50m"]) is False


# ─── 5. Separate Storage Tests ───────────────────────────────────────────────

class TestGISStorage:
    """Verify clean separation between raw and processed geospatial storage."""

    def test_save_and_load_processed_layer(self, tmp_path):
        geom = [LineString([(80.25, 13.05), (80.26, 13.06)])]
        gdf = gpd.GeoDataFrame({"segment_id": ["s1"], "highway": ["primary"]}, geometry=geom, crs=CRS_WGS84)

        saved_path = save_processed_layer(gdf, "test_roads", file_format="parquet", output_dir=tmp_path)
        assert saved_path.exists()
        assert saved_path.suffix == ".parquet"

        loaded = load_processed_layer("test_roads", file_format="parquet", input_dir=tmp_path)
        assert len(loaded) == 1
        assert loaded.iloc[0]["segment_id"] == "s1"
        assert loaded.crs.to_string() == CRS_WGS84


# ─── 6. Schema and Ingestion Contract Tests ──────────────────────────────────

class TestIngestionSchemas:
    """Verify documentation of schemas for upcoming municipal data feeds."""

    def test_expected_swd_columns_documented(self):
        assert "drain_id" in EXPECTED_SWD_COLUMNS
        assert "width_m" in EXPECTED_SWD_COLUMNS
        assert "depth_m" in EXPECTED_SWD_COLUMNS
        assert "basin" in EXPECTED_SWD_COLUMNS

    def test_expected_boundary_columns_documented(self):
        assert "zone_no" in EXPECTED_BOUNDARY_COLUMNS
        assert "zone_name" in EXPECTED_BOUNDARY_COLUMNS
