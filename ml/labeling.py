"""
Target variable labeling module for Chennai Flood Access Risk Mapper.
Operationalizes:
1. Observed real-world labels from historical floods / sensors.
2. Physics-based rule generation for synthetic scenario training samples.
3. Label noise injection for epistemic uncertainty.
4. Strict precedence of real observed evidence over synthetic assumptions.
"""
from typing import Optional, Union, Dict, Any, List
import numpy as np
import pandas as pd
import geopandas as gpd

from config.constants import DEPTH_THRESHOLDS_CM
from historical.models import HistoricalEvidenceType


# Default rainfall 24h threshold (mm) above which drainage capacity of highway class is challenged
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
    Generate target labels (road_inaccessible) based on real observed historical records.

    Rules:
        - Returns 1 if direct flood presence with OBSERVED evidence and confidence >= min_confidence.
        - Returns NaN (unknown) if unobserved or ambiguous (we never assume lack of report = dry).

    Parameters:
        roads_df: DataFrame with historical flood attributes.
        direct_flood_col: Boolean or numeric flag for flood inundation presence.
        evidence_col: Evidence tier ('OBSERVED', 'INFERRED', etc.).
        confidence_col: Historical confidence metric (0.0 to 1.0).
        min_confidence: Minimum confidence threshold to assign positive label.

    Returns:
        pd.Series: Float series with values 1.0 or np.nan.
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
    
    Physics Rule:
        A segment is labeled 1 (inaccessible) if:
        1. rainfall_24h >= threshold_by_highway_class
        AND
        2. At least ONE of:
           - local_elevation_context < -1.5m (topographic depression)
           - dist_to_water_m < 80.0m (adjacent to canal/river)
           - historical_flood_frequency > 0.25 (known recurring flood spot)
           - (slope < 1.0% and elevation_mean < 7.0m) (flat coastal lowlands)

    Noise Injection:
        Randomly flips 5-15% of negative synthetic samples in vulnerable areas to 1
        with flag `is_synthetic_noise=True` to simulate unmodeled local drain blockages.

    Preserves existing real OBSERVED labels without overwriting them.

    Parameters:
        df: Feature table with rainfall and topographic features.
        noise_flip_rate: Probability of flipping negative to positive in vulnerable areas.
        random_seed: Seed for reproducible noise generation.

    Returns:
        pd.DataFrame: Copy of df with columns:
            - 'road_inaccessible': 0.0 or 1.0
            - 'sample_source': 'SYNTHETIC' (or 'REAL' if preserved)
            - 'is_synthetic_noise': bool
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
            # Noise injection: in borderline situations with moderate rain, inject unmodeled drain blockage
            if rain[i] >= (thresh * 0.7) and (is_depression or is_flat_lowland):
                if rng.rand() < noise_flip_rate:
                    synthetic_labels[i] = 1.0
                    is_noise[i] = True

    # Check if df already had a real target label column
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
            # Retain real observed ground truth
            final_labels[i] = existing_labels[i]
            final_sources[i] = "REAL"
            is_noise[i] = False

    out["road_inaccessible"] = final_labels
    out["sample_source"] = final_sources
    out["is_synthetic_noise"] = is_noise

    return out
