"""
DEM / elevation data ingestion adapters.
Handles SRTM 30m, Copernicus 30m, and ALOS PALSAR 12.5m elevation GeoTIFFs.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import rasterio
from rasterio.windows import from_bounds
import numpy as np

from config.constants import CHENNAI_BBOX, CRS_WGS84


class ElevationAdapter:
    """Ingestion and metadata interface for elevation rasters."""

    def __init__(self, dem_path: str) -> None:
        self.dem_path = Path(dem_path)
        self.metadata: Dict[str, Any] = {
            "source_name": f"DEM:{self.dem_path.name}",
            "status": "uninitialized",
        }

    def validate_source(self) -> bool:
        """Check if DEM file exists and is readable by rasterio."""
        if not self.dem_path.exists() or not self.dem_path.is_file():
            return False
        try:
            with rasterio.open(self.dem_path) as src:
                return src.count >= 1
        except Exception:
            return False

    def get_metadata(self) -> Dict[str, Any]:
        """Inspect bounds, resolution, CRS, and nodata value of DEM."""
        if not self.validate_source():
            raise FileNotFoundError(
                f"DEM raster file not found or invalid: {self.dem_path}. "
                f"Expected format: Cloud-Optimized GeoTIFF (COG) or standard GeoTIFF (SRTM/Copernicus)."
            )

        with rasterio.open(self.dem_path) as src:
            bounds = src.bounds
            self.metadata.update({
                "crs": str(src.crs),
                "width": src.width,
                "height": src.height,
                "bounds": {
                    "left": bounds.left,
                    "bottom": bounds.bottom,
                    "right": bounds.right,
                    "top": bounds.top,
                },
                "resolution": src.res,
                "nodata": src.nodata,
                "driver": src.driver,
                "status": "ready",
            })
        return self.metadata

    def read_window_for_bbox(
        self,
        bbox: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, Any]:
        """Read elevation array and transform for specified bounding box."""
        if not self.validate_source():
            raise FileNotFoundError(f"DEM raster not available at {self.dem_path}")

        b = bbox or CHENNAI_BBOX
        with rasterio.open(self.dem_path) as src:
            window = from_bounds(
                b["west"], b["south"], b["east"], b["north"],
                transform=src.transform,
            )
            data = src.read(1, window=window)
            win_transform = src.window_transform(window)
            return data, win_transform

    def sample_point(self, lat: float, lon: float) -> float:
        """Sample single elevation value (meters) at lat, lon."""
        if not self.validate_source():
            raise FileNotFoundError(f"DEM raster not available at {self.dem_path}")

        with rasterio.open(self.dem_path) as src:
            vals = list(src.sample([(lon, lat)]))
            val = float(vals[0][0])
            if src.nodata is not None and val == src.nodata:
                return float("nan")
            return val
