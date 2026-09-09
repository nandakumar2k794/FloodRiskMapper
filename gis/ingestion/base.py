"""
Base geospatial data adapter interface.
Enables swappable data sources (OSM, municipal shapefiles, APIs, remote sensing).
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import geopandas as gpd


class BaseGeospatialAdapter(ABC):
    """Abstract base class for all geospatial data ingestion adapters."""

    def __init__(self, source_name: str, crs: str = "EPSG:4326") -> None:
        self.source_name = source_name
        self.default_crs = crs
        self.metadata: Dict[str, Any] = {
            "source_name": source_name,
            "crs": crs,
            "status": "uninitialized",
        }

    @abstractmethod
    def validate_source(self) -> bool:
        """Check if source (file or remote endpoint) is accessible."""
        pass

    @abstractmethod
    def load(self) -> gpd.GeoDataFrame:
        """Ingest raw geospatial data into a GeoPandas GeoDataFrame."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Return provenance and schema metadata."""
        return self.metadata

    def save_raw(self, output_path: Path, gdf: gpd.GeoDataFrame) -> Path:
        """Save raw ingested data to storage before preprocessing."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(output_path)
        self.metadata["raw_saved_to"] = str(output_path)
        return output_path
