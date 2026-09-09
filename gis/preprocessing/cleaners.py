"""
Domain-specific geospatial cleaning and missing data handling routines
for Chennai roads, waterways, drainage, and administrative boundaries.
"""
from typing import Dict, Any, Optional
import geopandas as gpd
import pandas as pd
import numpy as np

from gis.preprocessing.crs import ensure_crs, to_metric, to_geographic
from gis.preprocessing.geometry import validate_and_repair_geometries, compute_metric_dimensions
from config.constants import CRS_WGS84


DEFAULT_LANES_BY_HIGHWAY = {
    "motorway": 4,
    "motorway_link": 2,
    "trunk": 3,
    "trunk_link": 2,
    "primary": 2,
    "primary_link": 1,
    "secondary": 2,
    "secondary_link": 1,
    "tertiary": 2,
    "tertiary_link": 1,
    "residential": 1,
    "living_street": 1,
    "unclassified": 1,
    "service": 1,
}

DEFAULT_SPEED_BY_HIGHWAY = {
    "motorway": 80.0,
    "motorway_link": 50.0,
    "trunk": 60.0,
    "trunk_link": 40.0,
    "primary": 50.0,
    "primary_link": 35.0,
    "secondary": 40.0,
    "secondary_link": 30.0,
    "tertiary": 30.0,
    "tertiary_link": 25.0,
    "residential": 25.0,
    "living_street": 15.0,
    "unclassified": 25.0,
    "service": 20.0,
}


def clean_roads(
    gdf: gpd.GeoDataFrame,
    filter_drivable_only: bool = True,
) -> gpd.GeoDataFrame:
    """
    Standardize road network attributes and impute missing values.

    Handles:
    - Normalizing list-type tags (e.g. ['primary', 'trunk']) to single strings.
    - Imputing missing 'lanes' by road classification.
    - Imputing missing 'maxspeed' by road classification.
    - Standardizing oneway, bridge, and tunnel flags.
    - Calculating metric segment length in meters.
    """
    if gdf is None or gdf.empty:
        return gdf

    gdf = validate_and_repair_geometries(gdf)

    # Flatten list attributes common from OSMnx
    for col in ["highway", "name", "maxspeed", "lanes", "oneway", "bridge", "tunnel"]:
        if col in gdf.columns:
            gdf[col] = gdf[col].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)

    # Normalize highway classification
    if "highway" in gdf.columns:
        gdf["highway"] = gdf["highway"].astype(str).str.lower()
        if filter_drivable_only:
            drivable_classes = set(DEFAULT_LANES_BY_HIGHWAY.keys())
            gdf = gdf[gdf["highway"].isin(drivable_classes)].copy()

    # Impute missing lanes
    if "lanes" not in gdf.columns:
        gdf["lanes"] = np.nan

    def _impute_lanes(row):
        val = row.get("lanes")
        if pd.notnull(val):
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
        hw = str(row.get("highway", "")).lower()
        return float(DEFAULT_LANES_BY_HIGHWAY.get(hw, 1))

    gdf["lanes"] = gdf.apply(_impute_lanes, axis=1)

    # Impute missing maxspeed
    if "maxspeed" not in gdf.columns:
        gdf["maxspeed"] = np.nan

    def _impute_speed(row):
        val = row.get("maxspeed")
        if pd.notnull(val):
            try:
                # Handle '40 km/h' or '30 mph' strings
                num_str = "".join([c for c in str(val) if c.isdigit() or c == "."])
                if num_str:
                    return float(num_str)
            except (ValueError, TypeError):
                pass
        hw = str(row.get("highway", "")).lower()
        return float(DEFAULT_SPEED_BY_HIGHWAY.get(hw, 30.0))

    gdf["maxspeed"] = gdf.apply(_impute_speed, axis=1)

    # Clean boolean flags
    for flag_col in ["oneway", "bridge", "tunnel"]:
        if flag_col in gdf.columns:
            gdf[flag_col] = gdf[flag_col].apply(
                lambda x: True if str(x).lower() in ["true", "1", "yes"] else False
            )
        else:
            gdf[flag_col] = False

    # Ensure road segment has unique ID
    if "segment_id" not in gdf.columns:
        gdf["segment_id"] = [f"road_seg_{idx}" for idx in range(len(gdf))]

    # Ensure metric length
    if "length" in gdf.columns:
        gdf["length_m"] = gdf["length"].astype(float)
    else:
        gdf = compute_metric_dimensions(gdf, length_col="length_m")

    return gdf.reset_index(drop=True)


def clean_waterways(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Standardize surface water features (rivers, canals, lakes, reservoirs).
    Computes metric area for waterbody polygons and length for waterway lines.
    """
    if gdf is None or gdf.empty:
        return gdf

    gdf = validate_and_repair_geometries(gdf)

    if "name" in gdf.columns:
        gdf["name"] = gdf["name"].fillna("Unnamed Waterbody").astype(str)
    else:
        gdf["name"] = "Unnamed Waterbody"

    # Compute metric length or area
    gdf = compute_metric_dimensions(gdf, length_col="water_length_m", area_col="water_area_sqkm")
    return gdf.reset_index(drop=True)


def clean_boundaries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean administrative boundary polygons for Chennai zones and wards.
    Computes zone area in square kilometers and centroid coordinates.
    """
    if gdf is None or gdf.empty:
        return gdf

    gdf = validate_and_repair_geometries(gdf)
    gdf = compute_metric_dimensions(gdf, area_col="area_sqkm")

    # Compute centroid in WGS84 for labeling
    gdf_wgs84 = ensure_crs(gdf, target_crs=CRS_WGS84)
    gdf["centroid_lon"] = gdf_wgs84.geometry.centroid.x
    gdf["centroid_lat"] = gdf_wgs84.geometry.centroid.y

    return gdf.reset_index(drop=True)
