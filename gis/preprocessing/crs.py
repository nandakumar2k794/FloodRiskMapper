"""
Coordinate Reference System (CRS) transformations and validations.
Chennai spatial analysis uses:
- EPSG:4326 (WGS84 Geographic) for global positioning, GeoJSON, and display.
- EPSG:32644 (UTM Zone 44N) for metric distance, buffers, and area calculations.
"""
from typing import Optional
import geopandas as gpd
import pyproj

from config.constants import CRS_WGS84, CRS_UTM44N


def ensure_crs(
    gdf: gpd.GeoDataFrame,
    target_crs: str = CRS_WGS84,
    assume_crs_if_none: Optional[str] = CRS_WGS84,
) -> gpd.GeoDataFrame:
    """
    Ensure the GeoDataFrame has a valid CRS and transform to target_crs.

    Parameters:
        gdf: Input GeoDataFrame
        target_crs: Destination CRS (e.g. 'EPSG:4326' or 'EPSG:32644')
        assume_crs_if_none: CRS to assign if input has no CRS defined

    Returns:
        GeoDataFrame reprojected to target_crs
    """
    if gdf is None or gdf.empty:
        return gdf

    gdf = gdf.copy()

    if gdf.crs is None:
        if assume_crs_if_none is not None:
            gdf = gdf.set_crs(assume_crs_if_none)
        else:
            raise ValueError("Input GeoDataFrame has no CRS and assume_crs_if_none is None.")

    target_pyproj = pyproj.CRS.from_user_input(target_crs)
    current_pyproj = pyproj.CRS.from_user_input(gdf.crs)

    if current_pyproj != target_pyproj:
        gdf = gdf.to_crs(target_crs)

    return gdf


def to_metric(gdf: gpd.GeoDataFrame, utm_crs: str = CRS_UTM44N) -> gpd.GeoDataFrame:
    """Transform GeoDataFrame to projected metric CRS (UTM Zone 44N)."""
    return ensure_crs(gdf, target_crs=utm_crs)


def to_geographic(gdf: gpd.GeoDataFrame, geo_crs: str = CRS_WGS84) -> gpd.GeoDataFrame:
    """Transform GeoDataFrame to geographic WGS84 CRS."""
    return ensure_crs(gdf, target_crs=geo_crs)


def is_metric_crs(gdf: gpd.GeoDataFrame) -> bool:
    """Return True if GeoDataFrame is projected in a metric linear unit system."""
    if gdf is None or gdf.crs is None:
        return False
    return pyproj.CRS.from_user_input(gdf.crs).is_projected
