"""
GIS Module: OpenStreetMap road network extraction and spatial enrichment interface.
"""
from typing import Dict, Any, Optional


class RoadNetworkBuilder:
    """Interface for extracting OSM road graph and enriching road segments."""

    def __init__(self, bbox: Optional[Dict[str, float]] = None) -> None:
        self.bbox = bbox
        self._graph = None

    def fetch_network_from_osm(self, bbox: Dict[str, float], network_type: str = "drive") -> Any:
        """Download road network from OpenStreetMap within bounding box."""
        raise NotImplementedError("Phase 2 will implement OSM road network acquisition.")

    def enrich_segments(self, dem_path: str, hand_path: str) -> Any:
        """Enrich road segments with elevation, HAND, and drainage proximity metrics."""
        raise NotImplementedError("Phase 2 will implement road segment spatial enrichment.")
