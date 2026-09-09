"""
Terrain feature extraction module for road networks using Digital Elevation Models (DEM).
Extracts:
- mean elevation
- minimum elevation
- maximum elevation
- elevation range
- slope
- local elevation context (relative topographic position vs neighborhood)
"""
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.sample import sample_gen
from shapely.geometry import LineString, Point, MultiLineString

from gis.preprocessing.crs import ensure_crs, to_metric
from config.constants import CRS_WGS84, CRS_UTM44N


class TerrainFeatureExtractor:
    """
    Extracts road-level terrain features from Digital Elevation Model (DEM) rasters.
    Performs spatial sampling along road geometries and local neighborhood context extraction.
    """

    def __init__(self, dem_path: Optional[Union[str, Path]] = None, num_samples_per_segment: int = 5) -> None:
        self.dem_path = Path(dem_path) if dem_path else None
        self.num_samples_per_segment = max(2, num_samples_per_segment)

    def validate_dem(self) -> None:
        """Validate DEM raster availability and readability."""
        if not self.dem_path:
            raise FileNotFoundError(
                "DEM raster path is required for terrain feature extraction. "
                "Please provide a valid DEM GeoTIFF (e.g., SRTM 30m, Copernicus 30m, or ALOS 12.5m)."
            )
        if not self.dem_path.exists() or not self.dem_path.is_file():
            raise FileNotFoundError(
                f"DEM raster file not found at: {self.dem_path}. "
                "Terrain feature extraction requires a valid DEM raster file."
            )
        try:
            with rasterio.open(self.dem_path) as src:
                if src.count < 1:
                    raise ValueError(f"DEM raster has no valid bands: {self.dem_path}")
        except Exception as exc:
            raise FileNotFoundError(f"Failed to open DEM raster at {self.dem_path}: {exc}") from exc

    def extract_features(
        self,
        roads_gdf: gpd.GeoDataFrame,
        dem_path: Optional[Union[str, Path]] = None,
        context_buffer_meters: float = 50.0,
    ) -> gpd.GeoDataFrame:
        """
        Extract comprehensive road-level terrain features from DEM.

        Parameters:
            roads_gdf: GeoDataFrame containing road geometries.
            dem_path: Optional override for DEM raster path.
            context_buffer_meters: Radius in meters to sample surrounding neighborhood for context.

        Returns:
            GeoDataFrame with required columns:
              - 'road_id'
              - 'elevation' (mean elevation in meters)
              - 'slope' (topographic slope percentage)
              and detailed terrain metrics:
              - 'elevation_mean'
              - 'elevation_min'
              - 'elevation_max'
              - 'elevation_range'
              - 'local_elevation_context' (relative height vs surrounding neighborhood)
        """
        if roads_gdf is None or roads_gdf.empty:
            return gpd.GeoDataFrame(
                columns=[
                    "road_id",
                    "elevation",
                    "slope",
                    "elevation_mean",
                    "elevation_min",
                    "elevation_max",
                    "elevation_range",
                    "local_elevation_context",
                    "geometry",
                ],
                crs=CRS_WGS84,
            )

        if dem_path:
            self.dem_path = Path(dem_path)

        self.validate_dem()

        # Ensure WGS84 for coordinate sampling matching raster standard
        roads = ensure_crs(roads_gdf, CRS_WGS84).copy()

        # Ensure consistent road_id column
        if "road_id" not in roads.columns:
            if "segment_id" in roads.columns:
                roads["road_id"] = roads["segment_id"].astype(str)
            else:
                roads["road_id"] = [f"road_{idx}" for idx in range(len(roads))]

        # Ensure metric length for slope calculation
        if "length_m" not in roads.columns:
            roads_metric = to_metric(roads)
            roads["length_m"] = roads_metric.geometry.length

        # Collect sampling coordinates for all segments
        # 1. Profile points along the road geometry
        # 2. Left and Right neighborhood buffer points for local elevation context
        coords_along_roads: List[List[Tuple[float, float]]] = []
        context_sample_coords: List[List[Tuple[float, float]]] = []

        roads_metric = to_metric(roads)

        for i, row in roads.iterrows():
            geom = row.geometry
            metric_geom = roads_metric.geometry.iloc[i]
            
            # Sample points along linear geometry
            pts_along = self._sample_points_along_geom(geom, self.num_samples_per_segment)
            coords_along_roads.append([(p.x, p.y) for p in pts_along])

            # Sample local neighborhood points for local elevation context
            ctx_pts = self._sample_context_points(metric_geom, context_buffer_meters)
            coords_along_roads_ctx = ensure_crs(
                gpd.GeoDataFrame(geometry=ctx_pts, crs=CRS_UTM44N),
                CRS_WGS84,
            )
            context_sample_coords.append([(p.x, p.y) for p in coords_along_roads_ctx.geometry])

        # Flatten coordinate lists for bulk sampling with rasterio
        all_profile_pts = [pt for sublist in coords_along_roads for pt in sublist]
        all_context_pts = [pt for sublist in context_sample_coords for pt in sublist]

        with rasterio.open(self.dem_path) as src:
            nodata = src.nodata

            profile_vals = [float(v[0]) for v in src.sample(all_profile_pts)]
            context_vals = [float(v[0]) for v in src.sample(all_context_pts)]

            # Mask nodata
            if nodata is not None:
                profile_vals = [np.nan if v == nodata else v for v in profile_vals]
                context_vals = [np.nan if v == nodata else v for v in context_vals]

        # Reconstruct per-road statistics
        elev_means: List[float] = []
        elev_mins: List[float] = []
        elev_maxs: List[float] = []
        elev_ranges: List[float] = []
        slopes: List[float] = []
        local_contexts: List[float] = []

        idx_p = 0
        idx_c = 0
        for i, row in roads.iterrows():
            k_p = len(coords_along_roads[i])
            road_elevs = [v for v in profile_vals[idx_p : idx_p + k_p] if not np.isnan(v)]
            idx_p += k_p

            k_c = len(context_sample_coords[i])
            ctx_elevs = [v for v in context_vals[idx_c : idx_c + k_c] if not np.isnan(v)]
            idx_c += k_c

            if road_elevs:
                e_mean = float(np.mean(road_elevs))
                e_min = float(np.min(road_elevs))
                e_max = float(np.max(road_elevs))
                e_range = float(e_max - e_min)

                # Slope calculation: elevation change along length
                length = row.get("length_m", 1.0)
                if length > 0 and len(road_elevs) >= 2:
                    slope_val = float((abs(road_elevs[-1] - road_elevs[0]) / length) * 100.0)
                else:
                    slope_val = 0.0

                # Local elevation context: road height minus local neighborhood average
                # Negative = road is in a depression / low point; Positive = road is on an embankment
                if ctx_elevs:
                    ctx_mean = float(np.mean(ctx_elevs))
                    loc_context = float(e_mean - ctx_mean)
                else:
                    loc_context = 0.0
            else:
                e_mean = np.nan
                e_min = np.nan
                e_max = np.nan
                e_range = np.nan
                slope_val = np.nan
                loc_context = np.nan

            elev_means.append(e_mean)
            elev_mins.append(e_min)
            elev_maxs.append(e_max)
            elev_ranges.append(e_range)
            slopes.append(slope_val)
            local_contexts.append(loc_context)

        # Populate output feature table
        roads["elevation"] = elev_means
        roads["slope"] = slopes
        roads["elevation_mean"] = elev_means
        roads["elevation_min"] = elev_mins
        roads["elevation_max"] = elev_maxs
        roads["elevation_range"] = elev_ranges
        roads["local_elevation_context"] = local_contexts

        return roads

    @staticmethod
    def _sample_points_along_geom(geom: Any, num_points: int) -> List[Point]:
        """Generate equidistant sample points along a LineString or MultiLineString."""
        if isinstance(geom, Point):
            return [geom]
        elif isinstance(geom, LineString):
            if geom.length == 0:
                return [Point(geom.coords[0])]
            fractions = np.linspace(0.0, 1.0, num_points)
            return [geom.interpolate(f, normalized=True) for f in fractions]
        elif isinstance(geom, MultiLineString):
            pts = []
            for line in geom.geoms:
                pts.extend(TerrainFeatureExtractor._sample_points_along_geom(line, num_points))
            return pts[:num_points] if pts else [Point(0, 0)]
        else:
            return [geom.centroid]

    @staticmethod
    def _sample_context_points(metric_geom: Any, buffer_dist: float) -> List[Point]:
        """Sample points in the immediate neighborhood buffer around the segment."""
        centroid = metric_geom.centroid
        # Generate 4 cardinal offset points around centroid at buffer distance
        return [
            Point(centroid.x + buffer_dist, centroid.y),
            Point(centroid.x - buffer_dist, centroid.y),
            Point(centroid.x, centroid.y + buffer_dist),
            Point(centroid.x, centroid.y - buffer_dist),
        ]


def extract_road_terrain_features(
    roads_gdf: gpd.GeoDataFrame,
    dem_path: Union[str, Path],
    num_samples_per_segment: int = 5,
    context_buffer_meters: float = 50.0,
) -> gpd.GeoDataFrame:
    """
    Convenience function for extracting road-level terrain features from DEM.

    Returns GeoDataFrame with required columns:
      - 'road_id'
      - 'elevation' (mean elevation in meters)
      - 'slope' (topographic slope percentage)
      - 'elevation_mean'
      - 'elevation_min'
      - 'elevation_max'
      - 'elevation_range'
      - 'local_elevation_context'
    """
    extractor = TerrainFeatureExtractor(
        dem_path=dem_path,
        num_samples_per_segment=num_samples_per_segment,
    )
    return extractor.extract_features(
        roads_gdf=roads_gdf,
        context_buffer_meters=context_buffer_meters,
    )
