"""
Proximity analysis: calculates Euclidean distance from road segments to
rivers, canals, lakes, and stormwater drains in metric units (meters).
"""
import geopandas as gpd
import numpy as np
from shapely.ops import nearest_points

from gis.preprocessing.crs import to_metric, ensure_crs
from config.constants import CRS_WGS84, CRS_UTM44N


def calculate_distance_to_water(
    roads_gdf: gpd.GeoDataFrame,
    water_gdf: gpd.GeoDataFrame,
    max_distance_m: float = 5000.0,
) -> gpd.GeoDataFrame:
    """
    Calculate the shortest distance in meters from each road segment to the nearest water feature.
    Projects both layers to UTM Zone 44N for accurate metric distance.

    Enriches roads_gdf with:
      - 'dist_to_water_m': Shortest distance to any water body (meters)
      - 'near_water_50m': Boolean flag if road is within 50m of water
      - 'near_water_200m': Boolean flag if road is within 200m of water
    """
    if roads_gdf is None or roads_gdf.empty:
        return roads_gdf

    original_crs = str(roads_gdf.crs or CRS_WGS84)
    roads = to_metric(roads_gdf)

    if water_gdf is None or water_gdf.empty:
        roads["dist_to_water_m"] = max_distance_m
        roads["near_water_50m"] = False
        roads["near_water_200m"] = False
        return ensure_crs(roads, original_crs)

    water = to_metric(water_gdf)

    # Use GeoPandas sindex.nearest for efficient spatial indexing
    nearest_idx = water.sindex.nearest(roads.geometry, return_all=False)

    distances = []
    for road_geom_idx, water_geom_idx in zip(nearest_idx[0], nearest_idx[1]):
        r_geom = roads.geometry.iloc[road_geom_idx]
        w_geom = water.geometry.iloc[water_geom_idx]
        dist = r_geom.distance(w_geom)
        distances.append(min(float(dist), max_distance_m))

    roads["dist_to_water_m"] = distances
    roads["near_water_50m"] = roads["dist_to_water_m"] <= 50.0
    roads["near_water_200m"] = roads["dist_to_water_m"] <= 200.0

    return ensure_crs(roads, original_crs)


def calculate_distance_to_drains(
    roads_gdf: gpd.GeoDataFrame,
    drains_gdf: gpd.GeoDataFrame,
    max_distance_m: float = 1000.0,
) -> gpd.GeoDataFrame:
    """
    Calculate the shortest distance in meters from each road segment to the nearest stormwater drain.
    """
    if roads_gdf is None or roads_gdf.empty:
        return roads_gdf

    original_crs = str(roads_gdf.crs or CRS_WGS84)
    roads = to_metric(roads_gdf)

    if drains_gdf is None or drains_gdf.empty:
        roads["dist_to_drain_m"] = max_distance_m
        roads["has_adjacent_drain"] = False
        return ensure_crs(roads, original_crs)

    drains = to_metric(drains_gdf)
    nearest_idx = drains.sindex.nearest(roads.geometry, return_all=False)

    distances = []
    for road_geom_idx, drain_geom_idx in zip(nearest_idx[0], nearest_idx[1]):
        r_geom = roads.geometry.iloc[road_geom_idx]
        d_geom = drains.geometry.iloc[drain_geom_idx]
        dist = r_geom.distance(d_geom)
        distances.append(min(float(dist), max_distance_m))

    roads["dist_to_drain_m"] = distances
    # Roads within 15 meters typically have adjacent roadside drains
    roads["has_adjacent_drain"] = roads["dist_to_drain_m"] <= 15.0

    return ensure_crs(roads, original_crs)
