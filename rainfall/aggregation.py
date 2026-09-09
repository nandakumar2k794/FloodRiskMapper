"""
Temporal aggregation module for rainfall observations.
Computes standard rolling-window accumulations:
- rainfall_15min
- rainfall_1h
- rainfall_3h
- rainfall_6h
- rainfall_24h
"""
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from rainfall.models import (
    RainfallReading,
    RainfallAggregates,
    RainfallSourceType,
)
from rainfall.cleaning import readings_to_dataframe


def aggregate_station_readings(
    readings: List[RainfallReading],
    as_of: Optional[datetime] = None,
    source_type: Optional[RainfallSourceType] = None,
) -> RainfallAggregates:
    """
    Calculate rolling cumulative rainfall windows as of a specified reference timestamp.

    Parameters:
        readings: List of discrete rainfall readings for a single station.
        as_of: Evaluation reference timestamp (defaults to latest timestamp in readings).
        source_type: Override or inherited source type (REAL or SIMULATED).

    Returns:
        RainfallAggregates containing 15m, 1h, 3h, 6h, and 24h accumulations.
    """
    if not readings:
        return RainfallAggregates(
            rainfall_15min=0.0,
            rainfall_1h=0.0,
            rainfall_3h=0.0,
            rainfall_6h=0.0,
            rainfall_24h=0.0,
            source_type=source_type or RainfallSourceType.REAL,
            as_of_timestamp=as_of,
        )

    # Determine source type from readings if not provided
    eff_source = source_type or readings[0].source_type

    df = readings_to_dataframe(readings)
    if as_of is None:
        as_of = df["timestamp"].max()

    # Filter readings up to `as_of`
    df_past = df[df["timestamp"] <= as_of].copy()
    if df_past.empty:
        return RainfallAggregates(
            rainfall_15min=0.0,
            rainfall_1h=0.0,
            rainfall_3h=0.0,
            rainfall_6h=0.0,
            rainfall_24h=0.0,
            source_type=eff_source,
            as_of_timestamp=as_of,
        )

    # Window definitions
    t_15m = as_of - timedelta(minutes=15)
    t_1h = as_of - timedelta(hours=1)
    t_3h = as_of - timedelta(hours=3)
    t_6h = as_of - timedelta(hours=6)
    t_24h = as_of - timedelta(hours=24)

    sum_15m = float(df_past[df_past["timestamp"] > t_15m]["precipitation_mm"].sum())
    sum_1h = float(df_past[df_past["timestamp"] > t_1h]["precipitation_mm"].sum())
    sum_3h = float(df_past[df_past["timestamp"] > t_3h]["precipitation_mm"].sum())
    sum_6h = float(df_past[df_past["timestamp"] > t_6h]["precipitation_mm"].sum())
    sum_24h = float(df_past[df_past["timestamp"] > t_24h]["precipitation_mm"].sum())

    return RainfallAggregates(
        rainfall_15min=round(sum_15m, 2),
        rainfall_1h=round(sum_1h, 2),
        rainfall_3h=round(sum_3h, 2),
        rainfall_6h=round(sum_6h, 2),
        rainfall_24h=round(sum_24h, 2),
        source_type=eff_source,
        as_of_timestamp=as_of,
    )


def compute_rolling_aggregates_timeseries(
    readings: List[RainfallReading],
    freq: str = "15min",
) -> pd.DataFrame:
    """
    Generate continuous regular time series of cumulative rolling rainfall metrics.
    Useful for feeding downstream hydrological models over a multi-day storm event.
    """
    if not readings:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "precipitation_mm",
                "rainfall_15min",
                "rainfall_1h",
                "rainfall_3h",
                "rainfall_6h",
                "rainfall_24h",
                "source_type",
            ]
        )

    df = readings_to_dataframe(readings)
    source_type_val = df["source_type"].iloc[0] if not df.empty else "REAL"

    # Resample to regular grid
    df = df.set_index("timestamp").resample(freq).sum(numeric_only=True)
    df["precipitation_mm"] = df["precipitation_mm"].fillna(0.0)

    # 15min is current bucket (1 interval)
    df["rainfall_15min"] = df["precipitation_mm"]
    # 1 hour = 4 x 15min
    df["rainfall_1h"] = df["precipitation_mm"].rolling(window=4, min_periods=1).sum()
    # 3 hours = 12 x 15min
    df["rainfall_3h"] = df["precipitation_mm"].rolling(window=12, min_periods=1).sum()
    # 6 hours = 24 x 15min
    df["rainfall_6h"] = df["precipitation_mm"].rolling(window=24, min_periods=1).sum()
    # 24 hours = 96 x 15min
    df["rainfall_24h"] = df["precipitation_mm"].rolling(window=96, min_periods=1).sum()

    df["source_type"] = source_type_val
    return df.reset_index()
