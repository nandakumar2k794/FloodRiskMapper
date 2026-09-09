"""
Historical flood dataset ingestion module.
Ingests observed satellite SAR extents, municipal complaint points, and synthetic return-period zones.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import geopandas as gpd

from historical.models import (
    HistoricalEvidenceType,
    HistoricalFloodEvent,
)
from gis.preprocessing.crs import ensure_crs
from config.constants import CRS_WGS84


# Documented Major Chennai Flood Benchmark Events
CHENNAI_HISTORICAL_EVENTS = [
    HistoricalFloodEvent(
        event_id="EVENT_2015_DEC",
        name="December 2015 Chennai Deluge",
        year=2015,
        start_date="2015-11-30",
        end_date="2015-12-04",
        peak_24h_rainfall_mm=494.0,
        description="Historic Adyar/Cooum river breach & Chembarambakkam discharge flood.",
        evidence_type=HistoricalEvidenceType.OBSERVED,
    ),
    HistoricalFloodEvent(
        event_id="EVENT_2021_NOV",
        name="November 2021 Monsoon Inundation",
        year=2021,
        start_date="2021-11-06",
        end_date="2021-11-12",
        peak_24h_rainfall_mm=215.0,
        description="Severe urban waterlogging across T. Nagar, Velachery, and Pulianthope.",
        evidence_type=HistoricalEvidenceType.OBSERVED,
    ),
    HistoricalFloodEvent(
        event_id="EVENT_2023_MICHAUNG",
        name="Cyclone Michaung December 2023",
        year=2023,
        start_date="2023-12-03",
        end_date="2023-12-05",
        peak_24h_rainfall_mm=430.0,
        description="Widespread extreme urban flooding across North & South Chennai.",
        evidence_type=HistoricalEvidenceType.OBSERVED,
    ),
]


class HistoricalFloodIngestor:
    """
    Ingests and validates historical flood vector datasets from official or satellite sources.
    Strictly categorizes data by evidence type: OBSERVED, INFERRED, or SYNTHETIC.
    """

    def __init__(self, events: Optional[List[HistoricalFloodEvent]] = None) -> None:
        self.events = {e.event_id: e for e in (events or CHENNAI_HISTORICAL_EVENTS)}

    def get_registered_events(self) -> List[HistoricalFloodEvent]:
        """Return list of known documented Chennai flood events."""
        return list(self.events.values())

    def load_flood_dataset(
        self,
        file_path: Path,
        event_id: str,
        evidence_type: HistoricalEvidenceType = HistoricalEvidenceType.OBSERVED,
        default_confidence: float = 1.0,
    ) -> gpd.GeoDataFrame:
        """
        Load historical inundation polygons or points from GIS vector file.

        Parameters:
            file_path: Path to Shapefile, GeoJSON, GeoPackage, or Parquet.
            event_id: Associated flood event ID.
            evidence_type: OBSERVED, INFERRED, or SYNTHETIC.
            default_confidence: Confidence score between 0.0 and 1.0.
        """
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(
                f"Historical flood file not found: {file_path}. "
                "Please provide real historical flood extents (e.g., ISRO/Copernicus SAR polygons, GCC logs)."
            )

        suffix = path.suffix.lower()
        if suffix == ".parquet":
            gdf = gpd.read_parquet(path)
        elif suffix in [".geojson", ".json"]:
            import json
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "features" in data:
                gdf = gpd.GeoDataFrame.from_features(data["features"], crs=CRS_WGS84)
            else:
                gdf = gpd.read_file(path)
        else:
            gdf = gpd.read_file(path)

        gdf = ensure_crs(gdf, CRS_WGS84)

        # Standardize metadata columns
        gdf["event_id"] = event_id
        gdf["evidence_type"] = evidence_type.value
        if "confidence_score" not in gdf.columns:
            gdf["confidence_score"] = float(default_confidence)

        return gdf
