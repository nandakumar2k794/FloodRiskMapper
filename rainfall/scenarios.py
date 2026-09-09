"""
Rainfall scenario generator and simulation input interface.
Supports arbitrary rainfall scenarios (50mm, 75mm, 100mm, 150mm, 200mm, 250mm, custom).
All scenarios are explicitly marked with source_type=SIMULATED.
"""
from typing import Dict, Any, Optional, List, Union
import geopandas as gpd
import pandas as pd
import numpy as np

from rainfall.models import (
    RainfallAggregates,
    RainfallSourceType,
    ScenarioParameters,
)
from gis.preprocessing.crs import ensure_crs
from config.constants import CRS_WGS84


# Temporal distribution profile fractions: (15min, 1h, 3h, 6h, 24h)
# 15m, 1h, 3h, 6h accumulations relative to 24h total
DISTRIBUTION_PROFILES = {
    # Typical intense coastal convective storm (Chennai monsoon storm profile)
    "MONSOON_BURST": {
        "fraction_15min": 0.08,
        "fraction_1h": 0.25,
        "fraction_3h": 0.50,
        "fraction_6h": 0.75,
        "fraction_24h": 1.00,
    },
    # Steady prolonged depression / cyclone rain (e.g. Cyclone Michaung 2023)
    "PROLONGED_CYCLONE": {
        "fraction_15min": 0.04,
        "fraction_1h": 0.15,
        "fraction_3h": 0.35,
        "fraction_6h": 0.60,
        "fraction_24h": 1.00,
    },
    # Uniform synthetic rate
    "UNIFORM": {
        "fraction_15min": 0.25 / 24.0,
        "fraction_1h": 1.0 / 24.0,
        "fraction_3h": 3.0 / 24.0,
        "fraction_6h": 6.0 / 24.0,
        "fraction_24h": 1.00,
    },
}

# Standard benchmark design storm scenarios
STANDARD_SCENARIOS = {
    "50mm": {"total_mm": 50.0, "name": "50mm Moderate Monsoon Shower"},
    "75mm": {"total_mm": 75.0, "name": "75mm Heavy Rain Warning"},
    "100mm": {"total_mm": 100.0, "name": "100mm Very Heavy Rainfall Event"},
    "150mm": {"total_mm": 150.0, "name": "150mm Severe Cyclonic Inundation"},
    "200mm": {"total_mm": 200.0, "name": "200mm Extreme Cloudburst / Deluge"},
    "250mm": {"total_mm": 250.0, "name": "250mm Historic Extreme Event (2015/2023 Benchmark)"},
}


class RainfallScenarioBuilder:
    """
    Constructs simulated rainfall scenarios for what-if stress testing and evacuation planning.
    Always generates outputs stamped with RainfallSourceType.SIMULATED.
    """

    @classmethod
    def get_standard_scenarios(cls) -> Dict[str, Dict[str, Any]]:
        """Return list of predefined scenario names and total 24h accumulations."""
        return STANDARD_SCENARIOS

    @classmethod
    def build_aggregates_from_total(
        cls,
        total_24h_mm: float,
        distribution: str = "MONSOON_BURST",
    ) -> RainfallAggregates:
        """
        Derive 15min, 1h, 3h, 6h, and 24h simulated accumulations from total 24h depth.

        Parameters:
            total_24h_mm: Total 24-hour rainfall in millimeters (arbitrary positive float).
            distribution: Profile curve ('MONSOON_BURST', 'PROLONGED_CYCLONE', 'UNIFORM').
        """
        if total_24h_mm < 0.0:
            raise ValueError(f"Scenario rainfall must be non-negative. Received: {total_24h_mm}")

        profile = DISTRIBUTION_PROFILES.get(distribution, DISTRIBUTION_PROFILES["MONSOON_BURST"])

        r_24h = float(total_24h_mm)
        r_6h = round(r_24h * profile["fraction_6h"], 2)
        r_3h = round(r_24h * profile["fraction_3h"], 2)
        r_1h = round(r_24h * profile["fraction_1h"], 2)
        r_15m = round(r_24h * profile["fraction_15min"], 2)

        return RainfallAggregates(
            rainfall_15min=r_15m,
            rainfall_1h=r_1h,
            rainfall_3h=r_3h,
            rainfall_6h=r_6h,
            rainfall_24h=r_24h,
            source_type=RainfallSourceType.SIMULATED,
        )

    @classmethod
    def apply_scenario_to_roads(
        cls,
        roads_gdf: gpd.GeoDataFrame,
        total_24h_mm: float,
        scenario_name: Optional[str] = None,
        distribution: str = "MONSOON_BURST",
        hotspot: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        """
        Apply simulated rainfall scenario across all road segments in GeoDataFrame.

        Parameters:
            roads_gdf: Road network GeoDataFrame
            total_24h_mm: Arbitrary 24h precipitation in mm (e.g. 50, 75, 100, 150, 200, 250, or custom)
            scenario_name: Label for the simulated scenario
            distribution: Temporal distribution profile
            hotspot: Optional spatial gradient ('SOUTH_CHENNAI', 'NORTH_CHENNAI', None for uniform)
        """
        if roads_gdf is None or roads_gdf.empty:
            return roads_gdf

        roads = ensure_crs(roads_gdf, CRS_WGS84).copy()
        base_aggs = cls.build_aggregates_from_total(total_24h_mm, distribution=distribution)

        scen_label = scenario_name or f"SIMULATED_{int(total_24h_mm)}MM_{distribution}"

        # Calculate spatial factor per road segment if hotspot is specified
        if hotspot in ["SOUTH_CHENNAI", "NORTH_CHENNAI"]:
            from gis.preprocessing.crs import to_metric
            centroids = to_metric(roads).geometry.centroid.to_crs(CRS_WGS84)
            # Latitude threshold roughly dividing Chennai (Adyar River ~ 13.01 N)
            split_lat = 13.02
            if hotspot == "SOUTH_CHENNAI":
                # South gets 1.25x rainfall, North gets 0.85x
                factors = np.where(centroids.y <= split_lat, 1.25, 0.85)
            else:
                # North gets 1.25x rainfall, South gets 0.85x
                factors = np.where(centroids.y > split_lat, 1.25, 0.85)
        else:
            factors = np.ones(len(roads))

        roads["rainfall_15min"] = np.round(base_aggs.rainfall_15min * factors, 2)
        roads["rainfall_1h"] = np.round(base_aggs.rainfall_1h * factors, 2)
        roads["rainfall_3h"] = np.round(base_aggs.rainfall_3h * factors, 2)
        roads["rainfall_6h"] = np.round(base_aggs.rainfall_6h * factors, 2)
        roads["rainfall_24h"] = np.round(base_aggs.rainfall_24h * factors, 2)

        # Explicitly tag as SIMULATED rainfall
        roads["rainfall_source_type"] = RainfallSourceType.SIMULATED.value
        roads["scenario_name"] = scen_label

        return roads
