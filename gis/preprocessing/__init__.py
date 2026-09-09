"""
Geospatial preprocessing package: CRS handling, geometry validation, and cleaning.
"""
from gis.preprocessing.crs import ensure_crs, to_metric, to_geographic, is_metric_crs
from gis.preprocessing.geometry import (
    validate_and_repair_geometries,
    filter_by_bbox,
    compute_metric_dimensions,
)
from gis.preprocessing.cleaners import (
    clean_roads,
    clean_waterways,
    clean_boundaries,
    DEFAULT_LANES_BY_HIGHWAY,
    DEFAULT_SPEED_BY_HIGHWAY,
)

__all__ = [
    "ensure_crs",
    "to_metric",
    "to_geographic",
    "is_metric_crs",
    "validate_and_repair_geometries",
    "filter_by_bbox",
    "compute_metric_dimensions",
    "clean_roads",
    "clean_waterways",
    "clean_boundaries",
    "DEFAULT_LANES_BY_HIGHWAY",
    "DEFAULT_SPEED_BY_HIGHWAY",
]
