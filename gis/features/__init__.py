"""
GIS feature engineering package: spatial joins, proximity, and terrain feature extraction.
"""
from gis.features.spatial_joins import assign_zone_and_ward, detect_water_crossings
from gis.features.proximity import calculate_distance_to_water, calculate_distance_to_drains
from gis.features.elevation_sampling import sample_elevation_for_roads
from gis.features.terrain import TerrainFeatureExtractor, extract_road_terrain_features

__all__ = [
    "assign_zone_and_ward",
    "detect_water_crossings",
    "calculate_distance_to_water",
    "calculate_distance_to_drains",
    "sample_elevation_for_roads",
    "TerrainFeatureExtractor",
    "extract_road_terrain_features",
]
