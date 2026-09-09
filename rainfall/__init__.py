"""
Chennai Rainfall Data Pipeline Package.
Provides ingestion of REAL rainfall observations, cleaning, temporal aggregation,
missing-data handling, spatial association with road networks, and arbitrary scenario generation.
"""
from rainfall.models import (
    RainfallSourceType,
    RainfallStation,
    RainfallReading,
    RainfallAggregates,
    ScenarioParameters,
)
from rainfall.ingestion import RealRainfallIngestor, CHENNAI_MET_STATIONS
from rainfall.cleaning import clean_rainfall_readings, readings_to_dataframe
from rainfall.aggregation import (
    aggregate_station_readings,
    compute_rolling_aggregates_timeseries,
)
from rainfall.missing import impute_temporal_gaps, spatial_fallback_impute
from rainfall.spatial import associate_rainfall_with_roads
from rainfall.scenarios import (
    RainfallScenarioBuilder,
    STANDARD_SCENARIOS,
    DISTRIBUTION_PROFILES,
)

__all__ = [
    "RainfallSourceType",
    "RainfallStation",
    "RainfallReading",
    "RainfallAggregates",
    "ScenarioParameters",
    "RealRainfallIngestor",
    "CHENNAI_MET_STATIONS",
    "clean_rainfall_readings",
    "readings_to_dataframe",
    "aggregate_station_readings",
    "compute_rolling_aggregates_timeseries",
    "impute_temporal_gaps",
    "spatial_fallback_impute",
    "associate_rainfall_with_roads",
    "RainfallScenarioBuilder",
    "STANDARD_SCENARIOS",
    "DISTRIBUTION_PROFILES",
]
