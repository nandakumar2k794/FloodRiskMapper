"""
Spatial association of historical flood records with Chennai road segments.
Accounts for spatial uncertainty, positional reporting tolerance, and
separates OBSERVED, INFERRED, and SYNTHETIC evidence tiers.
"""
from typing import List, Dict, Any, Optional
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString

from historical.models import (
    HistoricalEvidenceType,
    HistoricalFloodEvent,
)
from gis.preprocessing.crs import ensure_crs, to_metric
from config.constants import CRS_WGS84, CRS_UTM44N


def associate_historical_floods_with_roads(
    roads_gdf: gpd.GeoDataFrame,
    flood_records_gdf: gpd.GeoDataFrame,
    total_recorded_events: int = 3,
    direct_tolerance_m: float = 20.0,
    uncertainty_buffer_m: float = 80.0,
) -> gpd.GeoDataFrame:
    """
    Associate multi-event historical flood extents and complaint points with road segments.

    Parameters:
        roads_gdf: Road network GeoDataFrame.
        flood_records_gdf: Historical inundation geometries with 'event_id' and 'evidence_type'.
        total_recorded_events: Baseline count of historical events analyzed for frequency calculation.
        direct_tolerance_m: Buffer distance (m) to consider a direct OBSERVED hit.
        uncertainty_buffer_m: Extended buffer distance (m) to capture INFERRED proximity flooding.

    Returns:
        roads_gdf enriched with historical flood features:
          - 'historical_flood_count'
          - 'historical_flood_frequency'
          - 'historical_inundation_presence'
          - 'observed_flood_count'
          - 'inferred_flood_count'
          - 'synthetic_flood_count'
          - 'min_dist_to_flood_record_m'
          - 'historical_confidence'
          - 'primary_evidence_type'
          - 'historical_data_available'
    """
    if roads_gdf is None or roads_gdf.empty:
        return roads_gdf

    roads = ensure_crs(roads_gdf, CRS_WGS84).copy()

    # Ensure road_id exists
    if "road_id" not in roads.columns:
        if "segment_id" in roads.columns:
            roads["road_id"] = roads["segment_id"].astype(str)
        else:
            roads["road_id"] = [f"road_{i}" for i in range(len(roads))]

    # Default values when no historical flood records are provided
    if flood_records_gdf is None or flood_records_gdf.empty:
        roads["historical_flood_count"] = 0
        roads["historical_flood_frequency"] = 0.0
        roads["historical_inundation_presence"] = False
        roads["observed_flood_count"] = 0
        roads["inferred_flood_count"] = 0
        roads["synthetic_flood_count"] = 0
        roads["min_dist_to_flood_record_m"] = float("inf")
        roads["historical_confidence"] = 0.0
        roads["primary_evidence_type"] = "NONE"
        roads["historical_data_available"] = False
        return roads

    floods = ensure_crs(flood_records_gdf, CRS_WGS84).copy()

    # Reproject both to metric UTM Zone 44N for accurate buffering and distance measurement
    roads_metric = to_metric(roads)
    floods_metric = to_metric(floods)

    # Ensure required columns in flood records
    if "evidence_type" not in floods_metric.columns:
        floods_metric["evidence_type"] = HistoricalEvidenceType.OBSERVED.value
    if "event_id" not in floods_metric.columns:
        floods_metric["event_id"] = "UNKNOWN_EVENT"

    # Separate flood records by evidence tier
    obs_floods = floods_metric[floods_metric["evidence_type"] == HistoricalEvidenceType.OBSERVED.value]
    inf_floods = floods_metric[floods_metric["evidence_type"] == HistoricalEvidenceType.INFERRED.value]
    syn_floods = floods_metric[floods_metric["evidence_type"] == HistoricalEvidenceType.SYNTHETIC.value]

    # Create spatial buffers for direct and inferred matching
    roads_direct_buf = roads_metric.copy()
    roads_direct_buf["geometry"] = roads_metric.geometry.buffer(direct_tolerance_m)

    roads_extended_buf = roads_metric.copy()
    roads_extended_buf["geometry"] = roads_metric.geometry.buffer(uncertainty_buffer_m)

    # Helper to count distinct events intersecting a buffered road layer
    def _count_events(roads_buf: gpd.GeoDataFrame, flood_subset: gpd.GeoDataFrame) -> Dict[str, set]:
        if flood_subset.empty:
            return {r_id: set() for r_id in roads["road_id"]}
        joined = gpd.sjoin(roads_buf, flood_subset[["event_id", "geometry"]], how="inner", predicate="intersects")
        if joined.empty:
            return {r_id: set() for r_id in roads["road_id"]}
        res = {r_id: set() for r_id in roads["road_id"]}
        for r_id, ev_id in zip(joined["road_id"], joined["event_id"]):
            res[r_id].add(ev_id)
        return res

    # 1. Direct Observed Hits
    obs_direct_events = _count_events(roads_direct_buf, obs_floods)
    # 2. Extended Inferred Proximity Hits
    obs_inferred_events = _count_events(roads_extended_buf, obs_floods)
    # 3. Explicit Inferred Layer Hits
    inf_layer_events = _count_events(roads_direct_buf, inf_floods)
    # 4. Synthetic Simulation Hits
    syn_layer_events = _count_events(roads_direct_buf, syn_floods)

    # Compute minimum distance to any flood record using spatial indexing
    nearest_flood_idx = floods_metric.sindex.nearest(roads_metric.geometry, return_all=False)
    min_distances: List[float] = []
    for r_idx, f_idx in zip(nearest_flood_idx[0], nearest_flood_idx[1]):
        r_geom = roads_metric.geometry.iloc[r_idx]
        f_geom = floods_metric.geometry.iloc[f_idx]
        min_distances.append(float(r_geom.distance(f_geom)))

    # Assemble feature lists
    hist_counts = []
    hist_freqs = []
    hist_presence = []
    obs_counts = []
    inf_counts = []
    syn_counts = []
    confidences = []
    primary_types = []

    denom_events = max(1, total_recorded_events)

    for i, r_id in enumerate(roads["road_id"]):
        d_obs = obs_direct_events.get(r_id, set())
        d_inf_prox = obs_inferred_events.get(r_id, set()) - d_obs
        d_inf_explicit = inf_layer_events.get(r_id, set())
        all_inf = d_inf_prox.union(d_inf_explicit)
        d_syn = syn_layer_events.get(r_id, set())

        n_obs = len(d_obs)
        n_inf = len(all_inf)
        n_syn = len(d_syn)

        # Combined unique flood events (observed + inferred + synthetic)
        all_events = d_obs.union(all_inf).union(d_syn)
        total_ev_count = len(all_events)

        dist_m = min_distances[i] if i < len(min_distances) else float("inf")

        # Inundation presence: True if segment had direct or inferred flooding
        has_presence = (n_obs > 0) or (n_inf > 0) or (n_syn > 0)

        # Determine primary evidence tier
        if n_obs > 0:
            prim_type = HistoricalEvidenceType.OBSERVED.value
            conf = 1.0
        elif n_inf > 0:
            prim_type = HistoricalEvidenceType.INFERRED.value
            # Confidence decays from 0.9 to 0.4 based on distance within buffer
            conf = max(0.4, min(0.9, 1.0 - (dist_m / uncertainty_buffer_m) * 0.6))
        elif n_syn > 0:
            prim_type = HistoricalEvidenceType.SYNTHETIC.value
            conf = 0.65
        else:
            prim_type = "NONE"
            # Crucial requirement: Absence of historical record is NOT proof of safety.
            # Confidence in "no flood" is high only if surrounded by comprehensive survey data,
            # otherwise low (0.0 to 0.2) reflecting observation gap.
            conf = 0.0

        freq = round(min(1.0, total_ev_count / denom_events), 3)

        hist_counts.append(total_ev_count)
        hist_freqs.append(freq)
        hist_presence.append(has_presence)
        obs_counts.append(n_obs)
        inf_counts.append(n_inf)
        syn_counts.append(n_syn)
        confidences.append(round(conf, 3))
        primary_types.append(prim_type)

    roads["historical_flood_count"] = hist_counts
    roads["historical_flood_frequency"] = hist_freqs
    roads["historical_inundation_presence"] = hist_presence
    roads["observed_flood_count"] = obs_counts
    roads["inferred_flood_count"] = inf_counts
    roads["synthetic_flood_count"] = syn_counts
    roads["min_dist_to_flood_record_m"] = [round(d, 1) for d in min_distances]
    roads["historical_confidence"] = confidences
    roads["primary_evidence_type"] = primary_types
    roads["historical_data_available"] = True

    return roads
