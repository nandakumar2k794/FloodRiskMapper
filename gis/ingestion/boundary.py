"""
Chennai administrative boundary ingestion adapters.
Covers Greater Chennai Corporation (GCC) limit, 15 administrative zones,
and 200 municipal wards.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import geopandas as gpd
import osmnx as ox

from gis.ingestion.base import BaseGeospatialAdapter
from config.constants import CHENNAI_BBOX, CRS_WGS84


EXPECTED_BOUNDARY_COLUMNS = [
    "zone_no",       # 1 to 15 (e.g. Zone 5: Royapuram, Zone 9: Teynampet)
    "zone_name",
    "ward_no",       # 1 to 200
    "geometry",
]


class BoundaryAdapter(BaseGeospatialAdapter):
    """Abstract adapter for administrative boundary layers."""
    pass


class OSMBoundaryAdapter(BoundaryAdapter):
    """Queries administrative boundary polygons for Chennai from OpenStreetMap."""

    def __init__(self, query: str = "Chennai, Tamil Nadu, India") -> None:
        super().__init__(source_name="OSM_Admin_Boundary", crs=CRS_WGS84)
        self.query = query

    def validate_source(self) -> bool:
        return bool(self.query and len(self.query.strip()) > 0)

    def load(self) -> gpd.GeoDataFrame:
        """Fetch boundary polygon of Chennai."""
        try:
            gdf = ox.geocode_to_gdf(self.query)
        except Exception:
            # Fallback to bounding box polygon if geocode service unavailable
            from shapely.geometry import box
            b = CHENNAI_BBOX
            geom = box(b["west"], b["south"], b["east"], b["north"])
            gdf = gpd.GeoDataFrame([{"name": "Chennai Bounding Box", "geometry": geom}], crs=CRS_WGS84)

        if gdf.crs is None:
            gdf = gdf.set_crs(self.default_crs)
        else:
            gdf = gdf.to_crs(self.default_crs)

        self.metadata["total_boundaries"] = len(gdf)
        self.metadata["status"] = "loaded"
        return gdf


class FileBoundaryAdapter(BoundaryAdapter):
    """
    Ingests official GCC 15-zone or 200-ward boundaries from local file.
    Expected schema:
      - zone_no: Integer (1-15)
      - zone_name: e.g. Thiruvottiyur, Manali, Madhavaram, Tondiarpet, Royapuram,
                   Thiru-Vi-Ka Nagar, Ambattur, Anna Nagar, Teynampet, Kodambakkam,
                   Valasaravakkam, Alandur, Adyar, Perungudi, Sholinganallur
      - ward_no: Integer (1-200) [optional if zone level]
      - geometry: Polygon or MultiPolygon
    """

    def __init__(self, file_path: str, crs: str = CRS_WGS84) -> None:
        super().__init__(source_name=f"Admin_File:{Path(file_path).name}", crs=crs)
        self.file_path = Path(file_path)

    def validate_source(self) -> bool:
        return self.file_path.exists() and self.file_path.is_file()

    def load(self) -> gpd.GeoDataFrame:
        if not self.validate_source():
            raise FileNotFoundError(
                f"Boundary file not found: {self.file_path}. "
                f"Expected format: GeoPackage or Shapefile containing GCC zones/wards."
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

        self.metadata["total_boundaries"] = len(gdf)
        self.metadata["status"] = "loaded"
        return gdf
