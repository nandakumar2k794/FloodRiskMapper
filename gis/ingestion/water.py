"""
Water bodies, rivers, and canals ingestion adapters for Chennai.
Captures key hydrological features: Adyar River, Cooum River, Buckingham Canal,
Pallikaranai Marshland, Chembarambakkam Lake, Puzhal Lake, and major tanks.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import geopandas as gpd
import osmnx as ox

from gis.ingestion.base import BaseGeospatialAdapter
from config.constants import CHENNAI_BBOX, CRS_WGS84


class WaterAdapter(BaseGeospatialAdapter):
    """Abstract adapter for surface water hydrology."""
    pass


class OSMWaterAdapter(WaterAdapter):
    """Ingests real water bodies, rivers, and canals from OpenStreetMap."""

    def __init__(self, bbox: Optional[Dict[str, float]] = None) -> None:
        super().__init__(source_name="OpenStreetMap_Water", crs=CRS_WGS84)
        self.bbox = bbox or CHENNAI_BBOX

    def validate_source(self) -> bool:
        b = self.bbox
        return (
            isinstance(b, dict)
            and b.get("north", 0) > b.get("south", 0)
            and b.get("east", 0) > b.get("west", 0)
        )

    def load(self) -> gpd.GeoDataFrame:
        """Query OSM for water bodies and waterways within Chennai bounding box."""
        b = self.bbox
        tags = {
            "natural": ["water", "wetland"],
            "waterway": ["river", "canal", "stream", "drain"],
            "water": ["lake", "reservoir", "pond", "basin"],
        }
        gdf = ox.features_from_bbox(
            bbox=(b["west"], b["south"], b["east"], b["north"]),
            tags=tags,
        )

        if gdf.empty:
            gdf = gpd.GeoDataFrame(columns=["name", "waterway", "natural", "water_type", "geometry"], crs=CRS_WGS84)
            self.metadata["total_features"] = 0
            self.metadata["status"] = "empty"
            return gdf

        if gdf.crs is None:
            gdf = gdf.set_crs(self.default_crs)
        else:
            gdf = gdf.to_crs(self.default_crs)

        # Categorize water type
        gdf["water_category"] = gdf.apply(self._categorize_feature, axis=1)

        self.metadata["total_features"] = len(gdf)
        self.metadata["categories"] = gdf["water_category"].value_counts().to_dict()
        self.metadata["status"] = "loaded"
        return gdf

    @staticmethod
    def _categorize_feature(row: Any) -> str:
        waterway = str(row.get("waterway", "")).lower()
        natural = str(row.get("natural", "")).lower()
        water = str(row.get("water", "")).lower()
        name = str(row.get("name", "")).lower()

        if waterway == "river" or "river" in name or "cooum" in name or "adyar" in name:
            return "RIVER"
        elif waterway == "canal" or "canal" in name or "buckingham" in name or "nullah" in name:
            return "CANAL"
        elif natural == "wetland" or "marsh" in name or "pallikaranai" in name:
            return "WETLAND"
        elif natural == "water" or water in ["lake", "reservoir", "basin", "pond"] or "lake" in name or "eri" in name:
            return "LAKE_RESERVOIR"
        else:
            return "OTHER_WATER"


class FileWaterAdapter(WaterAdapter):
    """Loads water body vector features from local file."""

    def __init__(self, file_path: str, crs: str = CRS_WGS84) -> None:
        super().__init__(source_name=f"File:{Path(file_path).name}", crs=crs)
        self.file_path = Path(file_path)

    def validate_source(self) -> bool:
        return self.file_path.exists() and self.file_path.is_file()

    def load(self) -> gpd.GeoDataFrame:
        if not self.validate_source():
            raise FileNotFoundError(
                f"Water body file not found: {self.file_path}. "
                f"Expected format: GeoPackage or Shapefile containing Chennai lakes/rivers/canals."
            )

        suffix = self.file_path.suffix.lower()
        if suffix == ".parquet":
            gdf = gpd.read_parquet(self.file_path)
        else:
            gdf = gpd.read_file(self.file_path)

        if gdf.crs is None:
            gdf = gdf.set_crs(self.default_crs)
        else:
            gdf = gdf.to_crs(self.default_crs)

        self.metadata["total_features"] = len(gdf)
        self.metadata["status"] = "loaded"
        return gdf
