"""
Road network ingestion adapters for Chennai.
Supports OpenStreetMap via OSMnx and local files (GeoPackage, GeoJSON, Shapefile).
"""
from abc import abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import geopandas as gpd
import osmnx as ox
import networkx as nx

from gis.ingestion.base import BaseGeospatialAdapter
from config.constants import CHENNAI_BBOX, CRS_WGS84


class RoadAdapter(BaseGeospatialAdapter):
    """Abstract adapter for road network datasets."""

    @abstractmethod
    def load_graph(self) -> nx.MultiDiGraph:
        """Load road network as a NetworkX MultiDiGraph."""
        pass


class OSMRoadAdapter(RoadAdapter):
    """Ingests real road networks from OpenStreetMap using OSMnx."""

    def __init__(
        self,
        bbox: Optional[Dict[str, float]] = None,
        network_type: str = "drive",
        custom_filter: Optional[str] = None,
    ) -> None:
        super().__init__(source_name="OpenStreetMap", crs=CRS_WGS84)
        self.bbox = bbox or CHENNAI_BBOX
        self.network_type = network_type
        self.custom_filter = custom_filter
        self._cached_graph: Optional[nx.MultiDiGraph] = None

    def validate_source(self) -> bool:
        """Check if bbox coordinates are valid geographic numbers."""
        b = self.bbox
        return (
            isinstance(b, dict)
            and b.get("north", 0) > b.get("south", 0)
            and b.get("east", 0) > b.get("west", 0)
        )

    def load_graph(self) -> nx.MultiDiGraph:
        """Fetch NetworkX MultiDiGraph from OSM within specified bounding box."""
        if self._cached_graph is not None:
            return self._cached_graph

        b = self.bbox
        graph = ox.graph_from_bbox(
            bbox=(b["west"], b["south"], b["east"], b["north"]),
            network_type=self.network_type,
            custom_filter=self.custom_filter,
            simplify=True,
        )
        self._cached_graph = graph
        self.metadata["num_nodes"] = len(graph.nodes)
        self.metadata["num_edges"] = len(graph.edges)
        return graph

    def load(self) -> gpd.GeoDataFrame:
        """Fetch OSM road network and return edges as a GeoPandas GeoDataFrame."""
        graph = self.load_graph()
        _, edges = ox.graph_to_gdfs(graph)

        if edges.crs is None:
            edges = edges.set_crs(self.default_crs)
        else:
            edges = edges.to_crs(self.default_crs)

        # Standardize segment identification
        edges = edges.reset_index()
        if "u" in edges.columns and "v" in edges.columns:
            edges["segment_id"] = (
                edges["u"].astype(str)
                + "_"
                + edges["v"].astype(str)
                + "_"
                + edges.get("key", 0).astype(str)
            )

        self.metadata["total_segments"] = len(edges)
        self.metadata["status"] = "loaded"
        return edges


class FileRoadAdapter(RoadAdapter):
    """Loads road network from local vector file (GeoPackage, GeoJSON, Shapefile, Parquet)."""

    def __init__(self, file_path: str, crs: str = CRS_WGS84) -> None:
        super().__init__(source_name=f"File:{Path(file_path).name}", crs=crs)
        self.file_path = Path(file_path)

    def validate_source(self) -> bool:
        return self.file_path.exists() and self.file_path.is_file()

    def load(self) -> gpd.GeoDataFrame:
        if not self.validate_source():
            raise FileNotFoundError(f"Road data file not found: {self.file_path}")

        suffix = self.file_path.suffix.lower()
        if suffix == ".parquet":
            gdf = gpd.read_parquet(self.file_path)
        else:
            gdf = gpd.read_file(self.file_path)

        if gdf.crs is None:
            gdf = gdf.set_crs(self.default_crs)
        else:
            gdf = gdf.to_crs(self.default_crs)

        self.metadata["total_segments"] = len(gdf)
        self.metadata["status"] = "loaded"
        return gdf

    def load_graph(self) -> nx.MultiDiGraph:
        """Convert loaded edges GeoDataFrame into a NetworkX MultiDiGraph."""
        gdf = self.load()
        return ox.graph_from_gdfs(gpd.GeoDataFrame(), gdf)
