"""
Target variable labeling module for ML preprocessing.
Supports real observed labels and physics-based synthetic scenario labels.
"""
from typing import Optional, Union, Dict, Any, List
import numpy as np
import pandas as pd


HIGHWAY_24H_RAIN_THRESHOLDS_MM = {
    "motorway": 120.0,
    "trunk": 110.0,
    "primary": 100.0,
    "secondary": 80.0,
    "tertiary": 70.0,
    "residential": 55.0,
    "unclassified": 50.0,
    "service": 45.0,
    "living_street": 40.0,
}
DEFAULT_24H_RAIN_THRESHOLD_MM = 60.0


def generate_real_labels(
    roads_df: pd.DataFrame,
    direct_flood_col: str = "historical_inundation_presence",
    evidence_col: str = "primary_evidence_type",
    confidence_col: str = "historical_confidence",
    min_confidence: float = 0.6,
) -> pd.Series:
    """
    Generate ground-truth accessibility labels based on real observed historical records.
    """
    n = len(roads_df)
    labels = np.full(n, np.nan, dtype=float)

    if n == 0:
        return pd.Series(labels, index=roads_df.index)

    has_flood = (
        roads_df[direct_flood_col].fillna(False).astype(bool).values
        if direct_flood_col in roads_df.columns
        else np.zeros(n, dtype=bool)
    )
    evidence = (
        roads_df[evidence_col].astype(str).str.upper().values
        if evidence_col in roads_df.columns
        else np.full(n, "NONE")
    )
    confidence = (
        roads_df[confidence_col].fillna(0.0).astype(float).values
        if confidence_col in roads_df.columns
        else np.zeros(n, dtype=float)
    )

    for i in range(n):
        if has_flood[i] and evidence[i] == "OBSERVED" and confidence[i] >= min_confidence:
            labels[i] = 1.0
        elif has_flood[i] and evidence[i] == "INFERRED" and confidence[i] >= 0.85:
            labels[i] = 1.0

    return pd.Series(labels, index=roads_df.index)


def generate_synthetic_scenario_labels(
    df: pd.DataFrame,
    rainfall_24h_col: str = "rainfall_24h",
    elevation_context_col: str = "local_elevation_context",
    dist_to_water_col: str = "dist_to_water_m",
    hist_freq_col: str = "historical_flood_frequency",
    elevation_mean_col: str = "elevation_mean",
    slope_col: str = "slope",
    highway_col: str = "highway",
    noise_flip_rate: float = 0.08,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic scenario training samples with physics-based labeling and noise injection.
    Preserves real observed labels without overwriting them.
    """
    out = df.copy()
    n = len(out)
    rng = np.random.RandomState(random_seed)

    rain = out[rainfall_24h_col].fillna(0.0).astype(float).values if rainfall_24h_col in out.columns else np.zeros(n)
    loc_ctx = out[elevation_context_col].fillna(0.0).astype(float).values if elevation_context_col in out.columns else np.zeros(n)
    dist_water = out[dist_to_water_col].fillna(9999.0).astype(float).values if dist_to_water_col in out.columns else np.full(n, 9999.0)
    hist_freq = out[hist_freq_col].fillna(0.0).astype(float).values if hist_freq_col in out.columns else np.zeros(n)
    elev_mean = out[elevation_mean_col].fillna(15.0).astype(float).values if elevation_mean_col in out.columns else np.full(n, 15.0)
    slopes = out[slope_col].fillna(2.0).astype(float).values if slope_col in out.columns else np.full(n, 2.0)
    highways = out[highway_col].astype(str).values if highway_col in out.columns else np.full(n, "residential")

    synthetic_labels = np.zeros(n, dtype=float)
    is_noise = np.zeros(n, dtype=bool)

    for i in range(n):
        hw = highways[i].lower()
        thresh = HIGHWAY_24H_RAIN_THRESHOLDS_MM.get(hw, DEFAULT_24H_RAIN_THRESHOLD_MM)

        # Physics evaluation
        is_heavy_rain = rain[i] >= thresh
        is_depression = loc_ctx[i] < -1.5
        is_near_water = dist_water[i] < 80.0
        is_chronically_flooded = hist_freq[i] > 0.25
        is_flat_lowland = (slopes[i] < 1.0) and (elev_mean[i] < 7.0)

        vulnerable_spot = is_depression or is_near_water or is_chronically_flooded or is_flat_lowland

        if is_heavy_rain and vulnerable_spot:
            synthetic_labels[i] = 1.0
        else:
            synthetic_labels[i] = 0.0
            if rain[i] >= (thresh * 0.7) and (is_depression or is_flat_lowland):
                if rng.rand() < noise_flip_rate:
                    synthetic_labels[i] = 1.0
                    is_noise[i] = True

    existing_labels = (
        out["road_inaccessible"].values
        if "road_inaccessible" in out.columns
        else np.full(n, np.nan)
    )
    existing_sources = (
        out["sample_source"].values
        if "sample_source" in out.columns
        else np.full(n, "SYNTHETIC")
    )

    final_labels = synthetic_labels.copy()
    final_sources = np.full(n, "SYNTHETIC", dtype=object)

    for i in range(n):
        if not pd.isna(existing_labels[i]) and str(existing_sources[i]).upper() == "REAL":
            final_labels[i] = existing_labels[i]
            final_sources[i] = "REAL"
            is_noise[i] = False

    out["road_inaccessible"] = final_labels
    out["sample_source"] = final_sources
    out["is_synthetic_noise"] = is_noise

    return out
