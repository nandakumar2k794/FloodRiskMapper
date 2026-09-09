"""
Historical flood feature table generator.
Produces structured, standardized feature tables separating OBSERVED, INFERRED, and SYNTHETIC tiers.
"""
from typing import Optional, Union, List, Dict
from pathlib import Path
import geopandas as gpd
import pandas as pd

from historical.models import HistoricalEvidenceType, HistoricalRoadFeatures
from historical.association import associate_historical_floods_with_roads
from historical.ingestion import CHENNAI_HISTORICAL_EVENTS
from gis.storage import save_processed_layer


HISTORICAL_FEATURE_COLUMNS = [
    "road_id",
    "historical_flood_count",
    "historical_flood_frequency",
    "historical_inundation_presence",
    "observed_flood_count",
    "inferred_flood_count",
    "synthetic_flood_count",
    "min_dist_to_flood_record_m",
    "historical_confidence",
    "primary_evidence_type",
    "historical_data_available",
]


class HistoricalFloodFeaturePipeline:
    """
    End-to-end pipeline for generating road-level historical flood features.
    """

    def __init__(
        self,
        direct_tolerance_m: float = 20.0,
        uncertainty_buffer_m: float = 80.0,
        total_benchmark_events: int = 3,
    ) -> None:
        self.direct_tolerance_m = direct_tolerance_m
        self.uncertainty_buffer_m = uncertainty_buffer_m
        self.total_benchmark_events = total_benchmark_events or len(CHENNAI_HISTORICAL_EVENTS)

    def extract_features(
        self,
        roads_gdf: gpd.GeoDataFrame,
        flood_records_gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> gpd.GeoDataFrame:
        """
        Compute historical flood features across road segments.

        Parameters:
            roads_gdf: Road network GeoDataFrame.
            flood_records_gdf: Historical inundation vector layers.

        Returns:
            GeoDataFrame with standardized historical flood feature columns.
        """
        if roads_gdf is None or roads_gdf.empty:
            return gpd.GeoDataFrame(columns=HISTORICAL_FEATURE_COLUMNS + ["geometry"])

        enriched = associate_historical_floods_with_roads(
            roads_gdf=roads_gdf,
            flood_records_gdf=flood_records_gdf,
            total_recorded_events=self.total_benchmark_events,
            direct_tolerance_m=self.direct_tolerance_m,
            uncertainty_buffer_m=self.uncertainty_buffer_m,
        )

        return enriched

    def save_features(
        self,
        features_gdf: gpd.GeoDataFrame,
        layer_name: str = "historical_flood_features",
        file_format: str = "parquet",
    ) -> Path:
        """Save the processed historical flood features to processed storage."""
        return save_processed_layer(features_gdf, layer_name=layer_name, file_format=file_format)
