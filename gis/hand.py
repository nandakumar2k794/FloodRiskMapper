"""
GIS Module: Height Above Nearest Drainage (HAND) calculation interface.
"""
from typing import Optional
from pathlib import Path


class HANDCalculator:
    """Interface for calculating Height Above Nearest Drainage (HAND)."""

    def __init__(self, hand_raster_path: Optional[str] = None) -> None:
        self.hand_raster_path = Path(hand_raster_path) if hand_raster_path else None
        self._raster = None

    def calculate_hand(self, dem_path: str, drainage_vector_path: str, output_path: str) -> str:
        """Compute HAND raster from DEM and drainage vector network."""
        raise NotImplementedError("Phase 2 will implement HAND hydro-spatial processing.")

    def get_hand_value(self, lat: float, lon: float) -> float:
        """Query relative elevation above nearest drainage line in meters."""
        raise NotImplementedError("Phase 2 will implement HAND value point lookup.")
