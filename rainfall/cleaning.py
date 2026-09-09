"""
Rainfall data cleaning and quality control module.
Applies meteorological validation checks, handles negative values,
detects impossible spikes, and ensures clean timestamp indexing.
"""
from typing import List, Tuple
import pandas as pd
import numpy as np

from rainfall.models import RainfallReading, RainfallSourceType


# Physical meteorological bounds for Chennai coastal tropics
MAX_15MIN_PRECIP_MM = 120.0   # Extreme convective cloudburst limit
MAX_HOURLY_PRECIP_MM = 180.0   # Beyond theoretical maximum intensity


def clean_rainfall_readings(readings: List[RainfallReading]) -> List[RainfallReading]:
    """
    Validate, sanitize, and quality-control rainfall readings.

    Rules applied:
    1. Negative precipitation values are set to 0.0 with quality_flag='NEGATIVE_CORRECTED'.
    2. Unphysically high spikes (> MAX_15MIN_PRECIP_MM) are flagged 'SUSPECT_EXTREME'.
    3. Chronologically sort readings per station.
    4. Remove exact duplicate timestamps per station.
    """
    if not readings:
        return []

    cleaned: List[RainfallReading] = []
    seen = set()

    # Sort primarily by station_id and timestamp
    sorted_readings = sorted(readings, key=lambda r: (r.station_id, r.timestamp))

    for r in sorted_readings:
        key = (r.station_id, r.timestamp)
        if key in seen:
            continue
        seen.add(key)

        val = r.precipitation_mm
        flag = r.quality_flag

        if val < 0.0:
            val = 0.0
            flag = "NEGATIVE_CORRECTED"
        elif val > MAX_15MIN_PRECIP_MM:
            flag = "SUSPECT_EXTREME"

        cleaned.append(
            RainfallReading(
                station_id=r.station_id,
                timestamp=r.timestamp,
                precipitation_mm=val,
                source_type=r.source_type,
                quality_flag=flag,
            )
        )

    return cleaned


def readings_to_dataframe(readings: List[RainfallReading]) -> pd.DataFrame:
    """Convert list of RainfallReading models into a standardized clean pandas DataFrame."""
    if not readings:
        return pd.DataFrame(
            columns=["station_id", "timestamp", "precipitation_mm", "source_type", "quality_flag"]
        )

    data = [
        {
            "station_id": r.station_id,
            "timestamp": r.timestamp,
            "precipitation_mm": r.precipitation_mm,
            "source_type": r.source_type.value,
            "quality_flag": r.quality_flag,
        }
        for r in readings
    ]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(by=["station_id", "timestamp"]).reset_index(drop=True)
