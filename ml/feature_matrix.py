"""
Feature Matrix Assembly Module for Chennai Flood Access Risk Mapper.
Combines Tiers A through F:
- Tier A: Dynamic rainfall accumulation windows (15min, 1h, 3h, 6h, 24h)
- Tier B: Terrain & topographic metrics from DEM (elevation, slope, local context)
- Tier C: Hydrological proximity metrics (distance to water bodies, stormwater drains)
- Tier D: Historical flood frequency, counts, and evidence tiers
- Tier E: DrainWatch AI CCTV metrics (score, confidence, staleness, availability)
- Tier F: Road structural attributes (highway class, length, lanes, bridge, zone)
- Explicit interaction terms
"""
from typing import Optional, Union, Dict, Any, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

from config.constants import DRAINWATCH_CLASS_SCORES


# Core numeric and encoded features fed to the Stage 1 model
MODEL_FEATURE_NAMES = [
    # Tier A: Rainfall
    "rainfall_15min",
    "rainfall_1h",
    "rainfall_3h",
    "rainfall_6h",
    "rainfall_24h",
    # Tier B: Terrain
    "elevation_mean",
    "elevation_min",
    "elevation_max",
    "elevation_range",
    "slope",
    "local_elevation_context",
    # Tier C: Hydrology & Proximity
    "dist_to_water_m",
    "near_water_50m",
    "near_water_200m",
    "dist_to_drain_m",
    "has_adjacent_drain",
    "crosses_water",
    # Tier D: Historical
    "historical_flood_count",
    "historical_flood_frequency",
    "historical_inundation_presence",
    "observed_flood_count",
    "inferred_flood_count",
    "historical_confidence",
    # Tier F: Structural
    "lanes",
    "length_m",
    "bridge",
    "tunnel",
    # Explicit Interaction Terms
    "rainfall_24h_x_elevation_context",
    "rainfall_1h_x_dist_to_drain",
    "hist_freq_x_rainfall_24h",
]

# Categorical features
CATEGORICAL_FEATURES = [
    "highway",
    "primary_evidence_type",
    "zone_no",
]

# Tier E: DrainWatch Stage 2 features
DRAINWATCH_FEATURE_NAMES = [
    "drainwatch_score",
    "drainwatch_confidence",
    "drainwatch_data_age_hours",
    "drainwatch_available",
]


class FeatureMatrixBuilder:
    """
    Constructs standardized, leakage-free feature matrices from GIS, terrain,
    rainfall, historical, and computer vision data pipelines.
    """

    def __init__(self, default_fill_values: Optional[Dict[str, float]] = None) -> None:
        self.default_fills = default_fill_values or {
            "rainfall_15min": 0.0,
            "rainfall_1h": 0.0,
            "rainfall_3h": 0.0,
            "rainfall_6h": 0.0,
            "rainfall_24h": 0.0,
            "elevation_mean": 10.0,
            "elevation_min": 8.0,
            "elevation_max": 12.0,
            "elevation_range": 4.0,
            "slope": 1.5,
            "local_elevation_context": 0.0,
            "dist_to_water_m": 5000.0,
            "near_water_50m": 0.0,
            "near_water_200m": 0.0,
            "dist_to_drain_m": 1000.0,
            "has_adjacent_drain": 0.0,
            "crosses_water": 0.0,
            "historical_flood_count": 0.0,
            "historical_flood_frequency": 0.0,
            "historical_inundation_presence": 0.0,
            "observed_flood_count": 0.0,
            "inferred_flood_count": 0.0,
            "historical_confidence": 0.0,
            "lanes": 2.0,
            "length_m": 100.0,
            "bridge": 0.0,
            "tunnel": 0.0,
        }

    def assemble_static_road_features(
        self,
        roads_gdf: gpd.GeoDataFrame,
        terrain_df: Optional[pd.DataFrame] = None,
        proximity_df: Optional[pd.DataFrame] = None,
        historical_df: Optional[pd.DataFrame] = None,
    ) -> gpd.GeoDataFrame:
        """
        Merge static road features (Tiers B, C, D, F) into a unified GeoDataFrame.

        Parameters:
            roads_gdf: Base road network (Tier F).
            terrain_df: DEM-derived terrain features (Tier B).
            proximity_df: Hydrological proximity features (Tier C).
            historical_df: Historical flood summary features (Tier D).

        Returns:
            gpd.GeoDataFrame containing combined static features.
        """
        if roads_gdf is None or roads_gdf.empty:
            return gpd.GeoDataFrame()

        df = roads_gdf.copy()
        id_col = "road_id" if "road_id" in df.columns else "segment_id"
        if id_col not in df.columns:
            df["road_id"] = [f"road_{i}" for i in range(len(df))]
            id_col = "road_id"

        # Standardize ID column
        df["road_id"] = df[id_col]

        # Merge Terrain (Tier B) if provided separately
        if terrain_df is not None and not terrain_df.empty:
            t_id = "road_id" if "road_id" in terrain_df.columns else "segment_id"
            t_cols = [c for c in terrain_df.columns if c in [
                "elevation_mean", "elevation_min", "elevation_max",
                "elevation_range", "slope", "local_elevation_context"
            ]]
            if t_cols and t_id in terrain_df.columns:
                df = df.merge(
                    terrain_df[[t_id] + t_cols].drop_duplicates(subset=[t_id]),
                    left_on="road_id",
                    right_on=t_id,
                    how="left",
                    suffixes=("", "_terrain"),
                )

        # Merge Proximity (Tier C) if provided separately
        if proximity_df is not None and not proximity_df.empty:
            p_id = "road_id" if "road_id" in proximity_df.columns else "segment_id"
            p_cols = [c for c in proximity_df.columns if c in [
                "dist_to_water_m", "near_water_50m", "near_water_200m",
                "dist_to_drain_m", "has_adjacent_drain", "crosses_water",
                "intersecting_water_type"
            ]]
            if p_cols and p_id in proximity_df.columns:
                df = df.merge(
                    proximity_df[[p_id] + p_cols].drop_duplicates(subset=[p_id]),
                    left_on="road_id",
                    right_on=p_id,
                    how="left",
                    suffixes=("", "_prox"),
                )

        # Merge Historical (Tier D) if provided separately
        if historical_df is not None and not historical_df.empty:
            h_id = "road_id" if "road_id" in historical_df.columns else "segment_id"
            h_cols = [c for c in historical_df.columns if c in [
                "historical_flood_count", "historical_flood_frequency",
                "historical_inundation_presence", "observed_flood_count",
                "inferred_flood_count", "historical_confidence",
                "primary_evidence_type", "historical_data_available"
            ]]
            if h_cols and h_id in historical_df.columns:
                df = df.merge(
                    historical_df[[h_id] + h_cols].drop_duplicates(subset=[h_id]),
                    left_on="road_id",
                    right_on=h_id,
                    how="left",
                    suffixes=("", "_hist"),
                )

        return df

    def combine_features(
        self,
        static_roads_df: pd.DataFrame,
        rainfall_df: Optional[pd.DataFrame] = None,
        drainwatch_df: Optional[pd.DataFrame] = None,
        include_drainwatch: bool = True,
    ) -> pd.DataFrame:
        """
        Combine static features with dynamic rainfall (Tier A) and DrainWatch (Tier E).
        Computes non-linear interaction terms.

        Parameters:
            static_roads_df: Road DataFrame with Tiers B, C, D, F.
            rainfall_df: Dynamic rainfall windows per road or global scenario.
            drainwatch_df: Vision-derived drain metrics per road.
            include_drainwatch: Whether to populate Tier E columns.

        Returns:
            pd.DataFrame: Complete feature matrix.
        """
        if static_roads_df is None or static_roads_df.empty:
            return pd.DataFrame()

        df = static_roads_df.copy()
        id_col = "road_id" if "road_id" in df.columns else "segment_id"
        if id_col not in df.columns:
            df["road_id"] = [f"road_{i}" for i in range(len(df))]
            id_col = "road_id"
        df["road_id"] = df[id_col]

        # 1. Integrate Rainfall (Tier A)
        if rainfall_df is not None and not rainfall_df.empty:
            r_id = "road_id" if "road_id" in rainfall_df.columns else "segment_id"
            r_cols = [c for c in rainfall_df.columns if c in [
                "rainfall_15min", "rainfall_1h", "rainfall_3h",
                "rainfall_6h", "rainfall_24h", "rainfall_source_type"
            ]]
            if r_cols and r_id in rainfall_df.columns:
                df = df.merge(
                    rainfall_df[[r_id] + r_cols].drop_duplicates(subset=[r_id]),
                    left_on="road_id",
                    right_on=r_id,
                    how="left",
                    suffixes=("", "_rain"),
                )
            else:
                # Rainfall provided as a single uniform row (scenario broadcast)
                for col in ["rainfall_15min", "rainfall_1h", "rainfall_3h", "rainfall_6h", "rainfall_24h"]:
                    if col in rainfall_df.columns:
                        df[col] = float(rainfall_df[col].iloc[0])
                if "rainfall_source_type" in rainfall_df.columns:
                    df["rainfall_source_type"] = str(rainfall_df["rainfall_source_type"].iloc[0])

        # Fill any missing rainfall with 0.0
        for r_c in ["rainfall_15min", "rainfall_1h", "rainfall_3h", "rainfall_6h", "rainfall_24h"]:
            if r_c not in df.columns:
                df[r_c] = 0.0
            else:
                df[r_c] = df[r_c].fillna(0.0).astype(float)

        if "rainfall_source_type" not in df.columns:
            df["rainfall_source_type"] = "SIMULATED"

        # 2. Integrate DrainWatch (Tier E)
        if include_drainwatch:
            if drainwatch_df is not None and not drainwatch_df.empty:
                d_id = "road_id" if "road_id" in drainwatch_df.columns else "segment_id"
                d_cols = [c for c in drainwatch_df.columns if c in [
                    "drainwatch_score", "drainwatch_observation_class",
                    "drainwatch_confidence", "drainwatch_data_age_hours"
                ]]
                if d_cols and d_id in drainwatch_df.columns:
                    df = df.merge(
                        drainwatch_df[[d_id] + d_cols].drop_duplicates(subset=[d_id]),
                        left_on="road_id",
                        right_on=d_id,
                        how="left",
                        suffixes=("", "_dw"),
                    )

            if "drainwatch_available" not in df.columns:
                if "drainwatch_score" in df.columns:
                    df["drainwatch_available"] = df["drainwatch_score"].notna()
                else:
                    df["drainwatch_available"] = False
                    df["drainwatch_score"] = np.nan

            if "drainwatch_confidence" not in df.columns:
                df["drainwatch_confidence"] = np.nan
            if "drainwatch_data_age_hours" not in df.columns:
                df["drainwatch_data_age_hours"] = np.nan
            if "drainwatch_observation_class" not in df.columns:
                df["drainwatch_observation_class"] = None

        # 3. Ensure all Tier B, C, D, F columns exist with fallbacks
        for col, default_val in self.default_fills.items():
            if col not in df.columns:
                df[col] = default_val
            else:
                df[col] = df[col].fillna(default_val)

        # Ensure boolean types are converted to numeric 0.0/1.0
        bool_cols = ["near_water_50m", "near_water_200m", "has_adjacent_drain", "crosses_water", "bridge", "tunnel"]
        for bc in bool_cols:
            if bc in df.columns:
                df[bc] = df[bc].astype(float)

        # 4. Compute Explicit Non-Linear Interaction Terms
        df["rainfall_24h_x_elevation_context"] = df["rainfall_24h"] * df["local_elevation_context"]
        df["rainfall_1h_x_dist_to_drain"] = df["rainfall_1h"] * df["dist_to_drain_m"]
        df["hist_freq_x_rainfall_24h"] = df["historical_flood_frequency"] * df["rainfall_24h"]

        # Ensure categorical columns are properly populated
        if "highway" not in df.columns:
            df["highway"] = "residential"
        else:
            df["highway"] = df["highway"].fillna("residential").astype(str)

        if "primary_evidence_type" not in df.columns:
            df["primary_evidence_type"] = "NONE"
        else:
            df["primary_evidence_type"] = df["primary_evidence_type"].fillna("NONE").astype(str)

        if "zone_no" not in df.columns:
            df["zone_no"] = "ZONE_UNKNOWN"
        else:
            df["zone_no"] = df["zone_no"].fillna("ZONE_UNKNOWN").astype(str)

        return df

    def get_feature_matrix(
        self,
        df: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        encode_categoricals: bool = True,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extract numeric feature matrix X ready for model training or inference.

        Parameters:
            df: DataFrame containing all assembled features.
            feature_names: List of column names to extract (defaults to MODEL_FEATURE_NAMES).
            encode_categoricals: If True, one-hot encodes CATEGORICAL_FEATURES.

        Returns:
            Tuple[pd.DataFrame, List[str]]: Feature matrix X and list of active feature names.
        """
        active_features = list(feature_names or MODEL_FEATURE_NAMES)
        
        # Verify required features exist in df
        missing = [f for f in active_features if f not in df.columns]
        if missing:
            raise ValueError(f"Feature matrix is missing required columns: {missing}")

        X = df[active_features].copy()

        if encode_categoricals:
            for cat_col in CATEGORICAL_FEATURES:
                if cat_col in df.columns:
                    # One-hot encode with prefix
                    dummies = pd.get_dummies(df[cat_col], prefix=cat_col, dtype=float)
                    X = pd.concat([X, dummies], axis=1)

        return X, list(X.columns)
