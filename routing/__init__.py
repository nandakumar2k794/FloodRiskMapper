"""
Inundation-Aware and Flood-Safe Routing package for Chennai.
"""
from routing.graph import (
    RoadGraphManager,
    compute_flood_cost,
    haversine_distance_m,
    DEFAULT_SPEED_LIMITS_KPH,
)
from routing.router import (
    FloodAwareRouter,
    RouteDetails,
    RouteComparison,
)

# Compatibility alias
FloodRiskRouter = FloodAwareRouter

__all__ = [
    "RoadGraphManager",
    "FloodAwareRouter",
    "FloodRiskRouter",
    "RouteDetails",
    "RouteComparison",
    "compute_flood_cost",
    "haversine_distance_m",
    "DEFAULT_SPEED_LIMITS_KPH",
]
