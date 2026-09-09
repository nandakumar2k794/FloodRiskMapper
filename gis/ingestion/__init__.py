"""
GIS Ingestion package. Swappable adapters for roads, water, drainage, boundaries, and DEM.
"""
from gis.ingestion.base import BaseGeospatialAdapter
from gis.ingestion.roads import RoadAdapter, OSMRoadAdapter, FileRoadAdapter
from gis.ingestion.water import WaterAdapter, OSMWaterAdapter, FileWaterAdapter
from gis.ingestion.drainage import (
    DrainageAdapter,
    FileDrainageAdapter,
    OSMDrainageAdapter,
    EXPECTED_SWD_COLUMNS,
)
from gis.ingestion.boundary import (
    BoundaryAdapter,
    OSMBoundaryAdapter,
    FileBoundaryAdapter,
    EXPECTED_BOUNDARY_COLUMNS,
)
from gis.ingestion.elevation import ElevationAdapter

__all__ = [
    "BaseGeospatialAdapter",
    "RoadAdapter",
    "OSMRoadAdapter",
    "FileRoadAdapter",
    "WaterAdapter",
    "OSMWaterAdapter",
    "FileWaterAdapter",
    "DrainageAdapter",
    "FileDrainageAdapter",
    "OSMDrainageAdapter",
    "EXPECTED_SWD_COLUMNS",
    "BoundaryAdapter",
    "OSMBoundaryAdapter",
    "FileBoundaryAdapter",
    "EXPECTED_BOUNDARY_COLUMNS",
    "ElevationAdapter",
]
