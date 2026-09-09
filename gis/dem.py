"""
GIS Module: DEM and elevation processing interface.
"""
from typing import Optional, Tuple
from pathlib import Path


class DEMProcessor:
    """Interface for digital elevation model (DEM) and topographic analysis."""

    def __init__(self, dem_path: Optional[str] = None) -> None:
        self.dem_path = Path(dem_path) if dem_path else None
        self._raster = None

    def load_dem(self, dem_path: str) -> None:
        """Load DEM raster from GeoTIFF."""
        raise NotImplementedError("Phase 2 will implement DEM raster loading.")

    def compute_slope(self) -> None:
        """Calculate topographic slope raster."""
        raise NotImplementedError("Phase 2 will implement slope computation.")

    def compute_twi(self) -> None:
        """Calculate Topographic Wetness Index (TWI)."""
        raise NotImplementedError("Phase 2 will implement TWI calculation.")

    def get_elevation_at(self, lat: float, lon: float) -> float:
        """Query elevation (meters) at specified geographic coordinate."""
        raise NotImplementedError("Phase 2 will implement elevation coordinate sampling.")
