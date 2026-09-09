"""
Unit tests for terrain feature extraction.
Verifies road-level terrain metrics (mean, min, max, range, slope, local context)
and proper handling when DEM is missing.
"""
import pytest
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

from config.constants import CRS_WGS84
from gis.features.terrain import TerrainFeatureExtractor, extract_road_terrain_features


@pytest.fixture
def sample_dem_raster(tmp_path):
    """Create a temporary GeoTIFF DEM raster with known elevation gradients."""
    dem_file = tmp_path / "test_dem.tif"

    # 100x100 grid covering roughly Chennai bbox area (lon 80.20 to 80.30, lat 13.00 to 13.10)
    width = 100
    height = 100
    res = 0.001  # ~110m per pixel
    transform = from_origin(80.20, 13.10, res, res)

    # Elevation surface: slopes upward from west (10m) to east (60m)
    # Plus a local depression (pond/quarry) in the middle
    x = np.linspace(10, 60, width)
    y = np.linspace(0, 0, height)
    xx, yy = np.meshgrid(x, y)
    elevation_data = (xx + yy).astype(np.float32)

    # Create a local depression at row 50, col 50
    elevation_data[45:55, 45:55] -= 15.0

    with rasterio.open(
        dem_file,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=CRS_WGS84,
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(elevation_data, 1)

    return dem_file


class TestTerrainFeatureExtraction:
    """Test terrain feature extraction from DEM."""

    def test_missing_dem_raises_error_without_inventing_values(self):
        """Pipeline must raise FileNotFoundError when DEM is unavailable."""
        extractor = TerrainFeatureExtractor(dem_path="non_existent_dem.tif")
        roads = gpd.GeoDataFrame(
            [{"road_id": "r1", "geometry": LineString([(80.22, 13.05), (80.24, 13.05)])}],
            crs=CRS_WGS84,
        )
        with pytest.raises(FileNotFoundError, match="Terrain feature extraction requires a valid DEM raster file"):
            extractor.extract_features(roads)

    def test_missing_dem_path_uninitialized(self):
        extractor = TerrainFeatureExtractor(dem_path=None)
        roads = gpd.GeoDataFrame(
            [{"road_id": "r1", "geometry": LineString([(80.22, 13.05), (80.24, 13.05)])}],
            crs=CRS_WGS84,
        )
        with pytest.raises(FileNotFoundError, match="DEM raster path is required"):
            extractor.extract_features(roads)

    def test_empty_roads_returns_empty_schema(self, sample_dem_raster):
        empty_roads = gpd.GeoDataFrame(columns=["road_id", "geometry"], crs=CRS_WGS84)
        result = extract_road_terrain_features(empty_roads, dem_path=sample_dem_raster)
        assert len(result) == 0
        assert "road_id" in result.columns
        assert "elevation" in result.columns
        assert "slope" in result.columns
        assert "elevation_mean" in result.columns

    def test_extracts_required_columns(self, sample_dem_raster):
        roads = gpd.GeoDataFrame(
            [
                {
                    "road_id": "road_annasalai",
                    "name": "Anna Salai",
                    "geometry": LineString([(80.22, 13.05), (80.26, 13.05)]),
                }
            ],
            crs=CRS_WGS84,
        )
        result = extract_road_terrain_features(roads, dem_path=sample_dem_raster)

        # Verify required columns
        assert "road_id" in result.columns
        assert "elevation" in result.columns
        assert "slope" in result.columns
        assert "elevation_mean" in result.columns
        assert "elevation_min" in result.columns
        assert "elevation_max" in result.columns
        assert "elevation_range" in result.columns
        assert "local_elevation_context" in result.columns

        row = result.iloc[0]
        assert row["road_id"] == "road_annasalai"
        assert 10.0 < row["elevation"] < 60.0
        assert row["elevation_max"] >= row["elevation_min"]
        assert row["elevation_range"] == pytest.approx(row["elevation_max"] - row["elevation_min"], rel=1e-3)
        assert row["slope"] >= 0.0

    def test_segment_id_fallback_to_road_id(self, sample_dem_raster):
        roads = gpd.GeoDataFrame(
            [
                {
                    "segment_id": "osm_seg_999",
                    "geometry": LineString([(80.21, 13.02), (80.23, 13.02)]),
                }
            ],
            crs=CRS_WGS84,
        )
        result = extract_road_terrain_features(roads, dem_path=sample_dem_raster)
        assert result.iloc[0]["road_id"] == "osm_seg_999"
        assert result.iloc[0]["elevation"] > 0

    def test_local_elevation_context_depression(self, sample_dem_raster):
        # Road passing directly through depression (row 50, col 50 = lon 80.25, lat 13.05)
        depression_road = gpd.GeoDataFrame(
            [
                {
                    "road_id": "dep_road",
                    "geometry": LineString([(80.248, 13.050), (80.252, 13.050)]),
                }
            ],
            crs=CRS_WGS84,
        )
        result = extract_road_terrain_features(depression_road, dem_path=sample_dem_raster, context_buffer_meters=150.0)
        row = result.iloc[0]

        # The road is in a 15m deep depression, so its local_elevation_context should be negative (lower than surroundings)
        assert row["local_elevation_context"] < 0
