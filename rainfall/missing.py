"""
Missing data imputation and data-quality handling for rainfall streams.
Provides temporal gap filling, spatial fallback imputation, and quality tracking.
"""
from typing import List, Dict, Optional, Tuple
from datetime import timedelta
import pandas as pd
import numpy as np

from rainfall.models import RainfallReading, RainfallStation, RainfallSourceType
from rainfall.cleaning import readings_to_dataframe


def impute_temporal_gaps(
    readings: List[RainfallReading],
    expected_freq: str = "15min",
    max_gap_hours: float = 3.0,
) -> List[RainfallReading]:
    """
    Detect missing intervals in a station's timeseries and impute values.

    Strategy:
    - Zero fill or linear interpolation for gaps up to max_gap_hours.
    - Tag imputed records with quality_flag='IMPUTED_TEMPORAL'.
    - Gaps larger than max_gap_hours are marked 'MISSING_UNRELIABLE'.
    """
    if not readings:
        return []

    df = readings_to_dataframe(readings)
    st_id = df["station_id"].iloc[0]
    source_type = df["source_type"].iloc[0]

    df = df.set_index("timestamp").sort_index()

    # Create complete target time index
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=expected_freq)
    df_reindexed = df.reindex(full_idx)
    df_reindexed["station_id"] = st_id
    df_reindexed["source_type"] = source_type

    # Identify missing values
    missing_mask = df_reindexed["precipitation_mm"].isna()

    # Interpolate small gaps
    max_intervals = int((max_gap_hours * 60) / 15)
    df_reindexed["precipitation_mm"] = df_reindexed["precipitation_mm"].interpolate(
        method="time",
        limit=max_intervals,
        limit_direction="both",
    )
    # Residual NaNs after limit are filled with 0.0 but flagged
    residual_mask = df_reindexed["precipitation_mm"].isna()
    df_reindexed["precipitation_mm"] = df_reindexed["precipitation_mm"].fillna(0.0)

    # Set quality flags
    df_reindexed["quality_flag"] = df_reindexed["quality_flag"].fillna("IMPUTED_TEMPORAL")
    df_reindexed.loc[residual_mask, "quality_flag"] = "MISSING_DEFAULTED_ZERO"

    imputed_readings: List[RainfallReading] = []
    for ts, row in df_reindexed.iterrows():
        imputed_readings.append(
            RainfallReading(
                station_id=st_id,
                timestamp=ts.to_pydatetime(),
                precipitation_mm=float(row["precipitation_mm"]),
                source_type=RainfallSourceType(row["source_type"]),
                quality_flag=str(row["quality_flag"]),
            )
        )

    return imputed_readings


def spatial_fallback_impute(
    target_station: RainfallStation,
    target_readings: List[RainfallReading],
    donor_stations: Dict[str, RainfallStation],
    donor_readings: Dict[str, List[RainfallReading]],
) -> List[RainfallReading]:
    """
    If a station completely fails or has significant outages, fall back to the nearest
    available operating station in Chennai.
    """
    if target_readings:
        return target_readings

    if not donor_readings:
        return []

    # Calculate geographic distance to all available donor stations
    t_lat, t_lon = target_station.latitude, target_station.longitude
    best_station_id = None
    min_dist_sq = float("inf")

    for s_id, s_info in donor_stations.items():
        if s_id == target_station.station_id or s_id not in donor_readings or not donor_readings[s_id]:
            continue
        dist_sq = (s_info.latitude - t_lat) ** 2 + (s_info.longitude - t_lon) ** 2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            best_station_id = s_id

    if not best_station_id:
        return []

    # Copy donor readings with quality tag reflecting spatial substitution
    fallback_readings: List[RainfallReading] = []
    for r in donor_readings[best_station_id]:
        fallback_readings.append(
            RainfallReading(
                station_id=target_station.station_id,
                timestamp=r.timestamp,
                precipitation_mm=r.precipitation_mm,
                source_type=r.source_type,
                quality_flag=f"SPATIAL_SUBSTITUTION_FROM_{best_station_id}",
            )
        )

    return fallback_readings
