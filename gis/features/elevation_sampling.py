"""
Elevation sampling for road segments using Digital Elevation Model rasters.
Samples elevation at segment start, midpoint, and end, and computes slope percentage.
"""
from pathlib import Path
from typing import Optional
import geopandas as gpd
import numpy as np
import rasterio

from gis.preprocessing.crs import ensure_crs
from config.constants import CRS_WGS84


def sample_elevation_for_roads(
    roads_gdf: gpd.GeoDataFrame,
    dem_path: str,
) -> gpd.GeoDataFrame:
    """
    Sample elevation in meters from a DEM raster GeoTIFF for each road segment.

    Adds:
      - 'elevation_m': Midpoint elevation in meters
      - 'elevation_start_m': Elevation at start node
      - 'elevation_end_m': Elevation at end node
      - 'slope_pct': Topographic slope percentage across segment
    """
    if roads_gdf is None or roads_gdf.empty:
        return roads_gdf

    path = Path(dem_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"DEM raster file not found: {dem_path}")

    roads = ensure_crs(roads_gdf, CRS_WGS84).copy()

    # Extract coordinates in (lon, lat) order expected by rasterio
    mid_coords = [(pt.x, pt.y) for pt in roads.geometry.centroid]
    start_coords = [(geom.coords[0][0], geom.coords[0][1]) for geom in roads.geometry]
    end_coords = [(geom.coords[-1][0], geom.coords[-1][1]) for geom in roads.geometry]

    with rasterio.open(path) as src:
        mid_samples = [float(val[0]) for val in src.sample(mid_coords)]
        start_samples = [float(val[0]) for val in src.sample(start_coords)]
        end_samples = [float(val[0]) for val in src.sample(end_coords)]

        nodata = src.nodata
        if nodata is not None:
            mid_samples = [np.nan if s == nodata else s for s in mid_samples]
            start_samples = [np.nan if s == nodata else s for s in start_samples]
            end_samples = [np.nan if s == nodata else s for s in end_samples]

    roads["elevation_m"] = mid_samples
    roads["elevation_start_m"] = start_samples
    roads["elevation_end_m"] = end_samples

    # Compute slope: abs(start - end) / length
    if "length_m" in roads.columns:
        lengths = roads["length_m"]
    else:
        from gis.preprocessing.geometry import compute_metric_dimensions
        roads = compute_metric_dimensions(roads)
        lengths = roads["length_m"]

    elev_diff = np.abs(np.array(start_samples) - np.array(end_samples))
    roads["slope_pct"] = np.where(lengths > 0, (elev_diff / lengths) * 100.0, 0.0)

    return roads
