"""
Chennai Historical Flood and Inundation Pipeline Package.
Provides ingestion, spatial matching, uncertainty handling, and feature extraction
for historical Chennai flood events (2015, 2021, 2023, GCC records).
"""
from historical.models import (
    HistoricalEvidenceType,
    HistoricalFloodEvent,
    HistoricalFloodRecord,
    HistoricalRoadFeatures,
)
from historical.ingestion import (
    HistoricalFloodIngestor,
    CHENNAI_HISTORICAL_EVENTS,
)
from historical.association import associate_historical_floods_with_roads
from historical.features import (
    HistoricalFloodFeaturePipeline,
    HISTORICAL_FEATURE_COLUMNS,
)

__all__ = [
    "HistoricalEvidenceType",
    "HistoricalFloodEvent",
    "HistoricalFloodRecord",
    "HistoricalRoadFeatures",
    "HistoricalFloodIngestor",
    "CHENNAI_HISTORICAL_EVENTS",
    "associate_historical_floods_with_roads",
    "HistoricalFloodFeaturePipeline",
    "HISTORICAL_FEATURE_COLUMNS",
]
