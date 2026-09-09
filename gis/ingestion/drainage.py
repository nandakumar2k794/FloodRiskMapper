"""
Stormwater drain (SWD) network ingestion adapters for Chennai.
Supports municipal drainage GIS layers and OpenStreetMap drain waterways.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import geopandas as gpd
import osmnx as ox

from gis.ingestion.base import BaseGeospatialAdapter
from config.constants import CHENNAI_BBOX, CRS_WGS84


EXPECTED_SWD_COLUMNS = [
    "drain_id",
    "drain_type",     # arterial, collector, micro, outfall
    "basin",          # Kosasthalaiyar, Cooum, Adyar, Kovalam
    "width_m",
    "depth_m",
    "material",       # RCC, masonry, earthen
    "geometry",
]


class DrainageAdapter(BaseGeospatialAdapter):
    """Abstract adapter for stormwater drain network."""
    pass


class FileDrainageAdapter(DrainageAdapter):
    """
    Ingests municipal stormwater drain network from local GIS file.
    Expected schema:
      - drain_id: Unique string identifier
      - drain_type: 'arterial', 'collector', 'micro', or 'outfall'
      - basin: Basin name (e.g. 'Adyar', 'Cooum', 'Kosasthalaiyar', 'Kovalam')
      - width_m: Top width of drain in meters (float)
      - depth_m: Depth of drain in meters (float)
      - material: Construction material ('RCC', 'masonry', 'earthen')
      - geometry: LineString or MultiLineString
    """

    def __init__(self, file_path: str, crs: str = CRS_WGS84) -> None:
        super().__init__(source_name=f"SWD_File:{Path(file_path).name}", crs=crs)
        self.file_path = Path(file_path)

    def validate_source(self) -> bool:
        return self.file_path.exists() and self.file_path.is_file()

    def load(self) -> gpd.GeoDataFrame:
        if not self.validate_source():
            raise FileNotFoundError(
                f"Stormwater drain file not found: {self.file_path}. "
                f"Expected format: Shapefile, GeoPackage, or Parquet with columns: {EXPECTED_SWD_COLUMNS}"
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

        # Standardize columns if present
        for col in ["width_m", "depth_m"]:
            if col in gdf.columns:
                gdf[col] = gdf[col].astype(float)

        self.metadata["total_drains"] = len(gdf)
        self.metadata["status"] = "loaded"
        return gdf


class OSMDrainageAdapter(DrainageAdapter):
    """Extracts mapped open drains and ditches from OpenStreetMap."""

    def __init__(self, bbox: Optional[Dict[str, float]] = None) -> None:
        super().__init__(source_name="OSM_Drainage", crs=CRS_WGS84)
        self.bbox = bbox or CHENNAI_BBOX

    def validate_source(self) -> bool:
        b = self.bbox
        return (
            isinstance(b, dict)
            and b.get("north", 0) > b.get("south", 0)
            and b.get("east", 0) > b.get("west", 0)
        )

    def load(self) -> gpd.GeoDataFrame:
        b = self.bbox
        tags = {"waterway": ["drain", "ditch"]}
        try:
            gdf = ox.features_from_bbox(
                bbox=(b["west"], b["south"], b["east"], b["north"]),
                tags=tags,
            )
        except Exception:
            gdf = gpd.GeoDataFrame(columns=["waterway", "geometry"], crs=CRS_WGS84)

        if gdf.empty:
            gdf = gpd.GeoDataFrame(columns=["waterway", "geometry"], crs=CRS_WGS84)

        if gdf.crs is None:
            gdf = gdf.set_crs(self.default_crs)
        else:
            gdf = gdf.to_crs(self.default_crs)

        gdf["drain_type"] = "osm_surface_drain"
        self.metadata["total_drains"] = len(gdf)
        self.metadata["status"] = "loaded"
        return gdf
