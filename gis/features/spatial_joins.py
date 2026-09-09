"""
Spatial joins module: links road network segments with administrative zones and water bodies.
"""
from typing import Optional
import geopandas as gpd
from shapely.geometry import Point

from gis.preprocessing.crs import ensure_crs
from config.constants import CRS_WGS84


def assign_zone_and_ward(
    roads_gdf: gpd.GeoDataFrame,
    boundaries_gdf: gpd.GeoDataFrame,
    predicate: str = "intersects",
) -> gpd.GeoDataFrame:
    """
    Perform spatial join assigning administrative zone and ward to each road segment.

    Parameters:
        roads_gdf: Road network GeoDataFrame
        boundaries_gdf: GCC zone or ward boundary polygons
        predicate: Spatial predicate ('intersects', 'within', 'centroid')

    Returns:
        roads_gdf enriched with boundary attributes (e.g. zone_no, zone_name, ward_no).
    """
    if roads_gdf is None or roads_gdf.empty:
        return roads_gdf
    if boundaries_gdf is None or boundaries_gdf.empty:
        roads_gdf = roads_gdf.copy()
        roads_gdf["zone_no"] = None
        roads_gdf["zone_name"] = None
        return roads_gdf

    roads = ensure_crs(roads_gdf, CRS_WGS84).copy()
    boundaries = ensure_crs(boundaries_gdf, CRS_WGS84).copy()

    # Retain only key boundary attributes to avoid collision
    boundary_cols = [c for c in ["zone_no", "zone_name", "ward_no"] if c in boundaries.columns]
    if not boundary_cols:
        boundary_cols = [c for c in boundaries.columns if c != "geometry"][:2]

    subset_boundaries = boundaries[boundary_cols + ["geometry"]]

    # Use metric projection for accurate segment centroid
    from gis.preprocessing.crs import to_metric
    roads_metric = to_metric(roads)
    roads["_join_pt"] = roads_metric.geometry.centroid.to_crs(CRS_WGS84)
    points_gdf = gpd.GeoDataFrame(
        roads[["segment_id", "_join_pt"]].rename(columns={"_join_pt": "geometry"}),
        crs=CRS_WGS84,
    )

    joined = gpd.sjoin(points_gdf, subset_boundaries, how="left", predicate="within")

    # Drop duplicate segment_ids if any point lies on boundary edge
    joined = joined.drop_duplicates(subset=["segment_id"])

    # Merge attributes back to roads
    attr_cols = [c for c in boundary_cols if c in joined.columns]
    merged = roads.merge(
        joined[["segment_id"] + attr_cols],
        on="segment_id",
        how="left",
    )

    if "_join_pt" in merged.columns:
        merged = merged.drop(columns=["_join_pt"])

    return merged


def detect_water_crossings(
    roads_gdf: gpd.GeoDataFrame,
    water_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Detect road segments that cross, touch, or traverse surface water features.
    Flags high-vulnerability points (bridges, causeways, low-level watercourse crossings).
    """
    if roads_gdf is None or roads_gdf.empty:
        return roads_gdf
    if water_gdf is None or water_gdf.empty:
        roads_gdf = roads_gdf.copy()
        roads_gdf["crosses_water"] = False
        roads_gdf["intersecting_water_type"] = None
        return roads_gdf

    roads = ensure_crs(roads_gdf, CRS_WGS84).copy()
    water = ensure_crs(water_gdf, CRS_WGS84).copy()

    roads["crosses_water"] = False
    roads["intersecting_water_type"] = None

    # Spatial join roads intersecting water
    water_cols = [c for c in ["water_category", "waterway", "name"] if c in water.columns]
    water_subset = water[water_cols + ["geometry"]]

    joined = gpd.sjoin(roads, water_subset, how="inner", predicate="intersects")
    if not joined.empty:
        crossing_ids = set(joined["segment_id"].unique())
        roads["crosses_water"] = roads["segment_id"].isin(crossing_ids)

        # Extract primary water category if available
        if "water_category" in joined.columns:
            cat_map = joined.groupby("segment_id")["water_category"].first().to_dict()
            roads["intersecting_water_type"] = roads["segment_id"].map(cat_map)

    return roads
