"""
Spatial association of rainfall metrics with Chennai road network segments.
Uses Inverse Distance Weighting (IDW) interpolation and nearest-gauge mapping.
"""
from typing import Dict, List, Optional
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point

from rainfall.models import RainfallStation, RainfallAggregates, RainfallSourceType
from gis.preprocessing.crs import ensure_crs, to_metric
from config.constants import CRS_WGS84, CRS_UTM44N


def associate_rainfall_with_roads(
    roads_gdf: gpd.GeoDataFrame,
    station_aggregates: Dict[str, RainfallAggregates],
    stations_info: Dict[str, RainfallStation],
    method: str = "idw",
    idw_power: float = 2.0,
) -> gpd.GeoDataFrame:
    """
    Interpolate and associate multi-interval rainfall fields across road segments.

    Parameters:
        roads_gdf: Road network segments GeoDataFrame
        station_aggregates: Map of station_id -> RainfallAggregates
        stations_info: Map of station_id -> RainfallStation (locations)
        method: 'idw' (Inverse Distance Weighting) or 'nearest' (Nearest Gauge)
        idw_power: IDW power parameter (typically 2.0 for precipitation)

    Returns:
        roads_gdf enriched with:
          - 'rainfall_15min'
          - 'rainfall_1h'
          - 'rainfall_3h'
          - 'rainfall_6h'
          - 'rainfall_24h'
          - 'rainfall_source_type' (REAL or SIMULATED)
          - 'nearest_rain_station'
    """
    if roads_gdf is None or roads_gdf.empty:
        return roads_gdf

    roads = ensure_crs(roads_gdf, CRS_WGS84).copy()

    # Filter stations that have both location and active aggregates
    valid_station_ids = [s_id for s_id in station_aggregates if s_id in stations_info]

    # If no stations available, set default zeros and return
    if not valid_station_ids:
        roads["rainfall_15min"] = 0.0
        roads["rainfall_1h"] = 0.0
        roads["rainfall_3h"] = 0.0
        roads["rainfall_6h"] = 0.0
        roads["rainfall_24h"] = 0.0
        roads["rainfall_source_type"] = RainfallSourceType.REAL.value
        roads["nearest_rain_station"] = "NONE"
        return roads

    # Check source type consistency across inputs
    first_st = station_aggregates[valid_station_ids[0]]
    source_type_val = first_st.source_type.value

    # Build station GeoDataFrame in metric CRS (UTM 44N) for accurate Euclidean distances
    st_pts = [
        Point(stations_info[s_id].longitude, stations_info[s_id].latitude)
        for s_id in valid_station_ids
    ]
    st_gdf = gpd.GeoDataFrame({"station_id": valid_station_ids}, geometry=st_pts, crs=CRS_WGS84)
    st_gdf_metric = to_metric(st_gdf)

    # Road segment centroids in metric CRS
    roads_metric = to_metric(roads)
    road_centroids = roads_metric.geometry.centroid

    st_coords_m = np.array([(pt.x, pt.y) for pt in st_gdf_metric.geometry])
    road_coords_m = np.array([(pt.x, pt.y) for pt in road_centroids])

    # Pre-extract arrays for all 5 intervals
    arr_15m = np.array([station_aggregates[s_id].rainfall_15min for s_id in valid_station_ids])
    arr_1h = np.array([station_aggregates[s_id].rainfall_1h for s_id in valid_station_ids])
    arr_3h = np.array([station_aggregates[s_id].rainfall_3h for s_id in valid_station_ids])
    arr_6h = np.array([station_aggregates[s_id].rainfall_6h for s_id in valid_station_ids])
    arr_24h = np.array([station_aggregates[s_id].rainfall_24h for s_id in valid_station_ids])

    # Compute pairwise Euclidean distance matrix (N_roads x M_stations) in meters
    # Using broadcasting: (N, 1, 2) - (1, M, 2) -> (N, M, 2)
    diff = road_coords_m[:, np.newaxis, :] - st_coords_m[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))  # (N, M)

    # Nearest station assignment for auditability
    nearest_idx = np.argmin(dist_matrix, axis=1)
    roads["nearest_rain_station"] = [valid_station_ids[idx] for idx in nearest_idx]

    if method == "nearest" or len(valid_station_ids) == 1:
        roads["rainfall_15min"] = arr_15m[nearest_idx]
        roads["rainfall_1h"] = arr_1h[nearest_idx]
        roads["rainfall_3h"] = arr_3h[nearest_idx]
        roads["rainfall_6h"] = arr_6h[nearest_idx]
        roads["rainfall_24h"] = arr_24h[nearest_idx]
    else:
        # Inverse Distance Weighting
        eps = 100.0  # 100m smoothing epsilon to prevent division by zero near stations
        weights = 1.0 / np.power(dist_matrix + eps, idw_power)
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        norm_weights = weights / weight_sums  # (N, M)

        roads["rainfall_15min"] = np.round(np.dot(norm_weights, arr_15m), 2)
        roads["rainfall_1h"] = np.round(np.dot(norm_weights, arr_1h), 2)
        roads["rainfall_3h"] = np.round(np.dot(norm_weights, arr_3h), 2)
        roads["rainfall_6h"] = np.round(np.dot(norm_weights, arr_6h), 2)
        roads["rainfall_24h"] = np.round(np.dot(norm_weights, arr_24h), 2)

    roads["rainfall_source_type"] = source_type_val
    return roads
