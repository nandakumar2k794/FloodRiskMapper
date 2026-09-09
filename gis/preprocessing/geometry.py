"""
Geospatial geometry validation, repair, and filtering utilities.
Uses Shapely 2.0+ and GeoPandas.
"""
from typing import Dict, Any, Optional
import geopandas as gpd
from shapely.geometry import box
from shapely.validation import make_valid
import shapely

from gis.preprocessing.crs import ensure_crs, to_metric, to_geographic
from config.constants import CHENNAI_BBOX, CRS_WGS84, CRS_UTM44N


def validate_and_repair_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Ensure all geometries in GeoDataFrame are non-null, non-empty, and topologically valid.

    1. Removes None and empty geometries.
    2. Runs make_valid() on invalid geometries (self-intersections, bowties).
    3. Resets index.
    """
    if gdf is None or gdf.empty:
        return gdf

    gdf = gdf.copy()

    # Drop null and empty
    valid_mask = ~gdf.geometry.isna() & ~gdf.geometry.is_empty
    gdf = gdf[valid_mask]

    # Identify invalid
    invalid_mask = ~gdf.geometry.is_valid
    if invalid_mask.any():
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].apply(make_valid)

    # Re-verify and drop any residual empties created during repair
    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty]
    return gdf.reset_index(drop=True)


def filter_by_bbox(
    gdf: gpd.GeoDataFrame,
    bbox: Optional[Dict[str, float]] = None,
    clip_geometries: bool = False,
) -> gpd.GeoDataFrame:
    """
    Filter features intersecting or within the specified bounding box.

    Parameters:
        gdf: Input GeoDataFrame
        bbox: dict with keys 'north', 'south', 'east', 'west'
        clip_geometries: If True, geometrically clips features to the bbox boundary.
    """
    if gdf is None or gdf.empty:
        return gdf

    b = bbox or CHENNAI_BBOX
    gdf = ensure_crs(gdf, target_crs=CRS_WGS84)
    bbox_poly = box(b["west"], b["south"], b["east"], b["north"])

    if clip_geometries:
        clipped = gpd.clip(gdf, bbox_poly)
        return validate_and_repair_geometries(clipped)
    else:
        # Spatial intersection filter
        mask = gdf.geometry.intersects(bbox_poly)
        return gdf[mask].reset_index(drop=True)


def compute_metric_dimensions(
    gdf: gpd.GeoDataFrame,
    length_col: str = "length_m",
    area_col: str = "area_sqkm",
) -> gpd.GeoDataFrame:
    """
    Calculate accurate metric length (meters) for LineStrings and area (sq km) for Polygons.
    Uses UTM Zone 44N projection for metric precision.
    """
    if gdf is None or gdf.empty:
        return gdf

    original_crs = gdf.crs or CRS_WGS84
    gdf_metric = to_metric(gdf)

    # Compute length for linear geometries
    linear_mask = gdf_metric.geometry.geom_type.isin(["LineString", "MultiLineString"])
    if linear_mask.any():
        gdf_metric.loc[linear_mask, length_col] = gdf_metric.loc[linear_mask, "geometry"].length

    # Compute area for polygonal geometries
    poly_mask = gdf_metric.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    if poly_mask.any():
        # m^2 / 1_000_000 = km^2
        gdf_metric.loc[poly_mask, area_col] = gdf_metric.loc[poly_mask, "geometry"].area / 1_000_000.0

    return ensure_crs(gdf_metric, target_crs=str(original_crs))
