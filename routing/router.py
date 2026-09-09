"""
Flood-Aware Road Routing Engine for Chennai.
Implements:
1. Normal shortest route (cost: travel_time).
2. Flood-aware route (cost: travel_time + quadratic flood risk penalty).
3. Dijkstra and A* pathfinding.
4. Route metrics: distance, travel time, risk score, high-risk segment count,
   additional distance, and additional travel time.
5. Side-by-side comparison between NORMAL ROUTE and FLOOD-SAFE ROUTE.
"""
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import networkx as nx
import numpy as np
from shapely.geometry import LineString, MultiLineString

from routing.graph import RoadGraphManager, haversine_distance_m


# Maximum speed in meters per second for A* admissible heuristic (80 km/h)
MAX_HEURISTIC_SPEED_MPS = 80.0 * 1000.0 / 3600.0


@dataclass
class RouteDetails:
    """Detailed metrics for an evaluated route."""
    route_type: str                  # "NORMAL" or "FLOOD_SAFE"
    nodes: List[str]
    segment_ids: List[str]
    distance_meters: float
    distance_km: float
    estimated_travel_time_seconds: float
    estimated_travel_time_minutes: float
    mean_risk_score: float
    max_risk_score: float
    number_of_high_risk_segments: int
    high_risk_segments: List[str]
    road_names: List[str]


@dataclass
class RouteComparison:
    """Side-by-side comparison between Normal and Flood-Safe routes."""
    normal_route: RouteDetails
    flood_safe_route: RouteDetails
    additional_distance_meters: float
    additional_distance_km: float
    additional_travel_time_seconds: float
    additional_travel_time_minutes: float
    risk_score_reduction: float
    high_risk_segments_avoided: int
    reroute_recommended: bool
    summary: str


class FloodAwareRouter:
    """
    Computes normal shortest and flood-safe alternative navigation routes
    on NetworkX road graphs.
    """

    def __init__(
        self,
        graph_manager: RoadGraphManager,
        algorithm: str = "dijkstra",
        high_risk_threshold: float = 50.0,
    ) -> None:
        self.graph_manager = graph_manager
        self.algorithm = algorithm.lower()
        self.high_risk_threshold = high_risk_threshold

    def _get_heuristic(self, target: str):
        """Admissible A* heuristic function estimating remaining travel time."""
        target_coords = self.graph_manager.node_coords.get(target)
        if not target_coords:
            return lambda u, v: 0.0

        t_lat, t_lon = target_coords

        def heuristic_fn(u: str, v: str = target) -> float:
            u_coords = self.graph_manager.node_coords.get(u)
            if not u_coords:
                return 0.0
            dist_m = haversine_distance_m(u_coords[0], u_coords[1], t_lat, t_lon)
            # Minimum time to reach target at maximum possible speed
            return dist_m / MAX_HEURISTIC_SPEED_MPS

        return heuristic_fn

    def _find_path(self, origin: str, destination: str, weight: str, avoid_critical: bool = False) -> List[str]:
        """Execute Dijkstra or A* pathfinding on the graph."""
        G = self.graph_manager.graph

        if avoid_critical:
            valid_edges = []
            if G.is_multigraph():
                for u, v, k, d in G.edges(keys=True, data=True):
                    if float(d.get("risk_score", 0.0)) < 75.0:
                        valid_edges.append((u, v, k))
            else:
                for u, v, d in G.edges(data=True):
                    if float(d.get("risk_score", 0.0)) < 75.0:
                        valid_edges.append((u, v))
            G_search = G.edge_subgraph(valid_edges)
        else:
            G_search = G

        if origin not in G_search:
            raise ValueError(f"Origin node '{origin}' not in road graph (or isolated by flood).")
        if destination not in G_search:
            raise ValueError(f"Destination node '{destination}' not in road graph (or isolated by flood).")
        if not nx.has_path(G_search, origin, destination):
            raise nx.NetworkXNoPath(f"No navigable path exists between '{origin}' and '{destination}'.")

        if self.algorithm == "astar":
            heuristic = self._get_heuristic(destination)
            return nx.astar_path(G_search, origin, destination, heuristic=heuristic, weight=weight)
        else:
            return nx.shortest_path(G_search, origin, destination, weight=weight)

    def extract_route_details(self, path_nodes: List[str], route_type: str = "NORMAL") -> RouteDetails:
        """
        Extract distance, travel time, and flood risk metrics along a sequence of nodes.
        """
        G = self.graph_manager.graph

        total_distance = 0.0
        total_time = 0.0
        risk_scores: List[float] = []
        high_risk_segs: List[str] = []
        segment_ids: List[str] = []
        road_names: List[str] = []

        for i in range(len(path_nodes) - 1):
            u = path_nodes[i]
            v = path_nodes[i + 1]

            edge = G[u][v]
            length = float(edge.get("length", 0.0))
            t_time = float(edge.get("travel_time", 0.0))
            risk = float(edge.get("risk_score", 0.0))
            seg_id = str(edge.get("segment_id", f"{u}_{v}"))
            name = str(edge.get("name", seg_id))

            total_distance += length
            total_time += t_time
            risk_scores.append(risk)
            segment_ids.append(seg_id)
            road_names.append(name)

            if risk >= self.high_risk_threshold:
                high_risk_segs.append(seg_id)

        mean_risk = float(np.mean(risk_scores)) if risk_scores else 0.0
        max_risk = float(np.max(risk_scores)) if risk_scores else 0.0

        return RouteDetails(
            route_type=route_type,
            nodes=path_nodes,
            segment_ids=segment_ids,
            distance_meters=round(total_distance, 1),
            distance_km=round(total_distance / 1000.0, 2),
            estimated_travel_time_seconds=round(total_time, 1),
            estimated_travel_time_minutes=round(total_time / 60.0, 1),
            mean_risk_score=round(mean_risk, 1),
            max_risk_score=round(max_risk, 1),
            number_of_high_risk_segments=len(high_risk_segs),
            high_risk_segments=high_risk_segs,
            road_names=road_names,
        )

    def find_normal_route(self, origin: str, destination: str) -> RouteDetails:
        """
        Calculate normal shortest route minimizing travel time.
        """
        path = self._find_path(origin, destination, weight="travel_time")
        return self.extract_route_details(path, route_type="NORMAL")

    def find_flood_safe_route(self, origin: str, destination: str) -> RouteDetails:
        """
        Calculate flood-safe route penalizing flooded and strictly avoiding CRITICAL high-risk segments.
        """
        path = self._find_path(origin, destination, weight="flood_cost", avoid_critical=True)
        return self.extract_route_details(path, route_type="FLOOD_SAFE")

    def compare_routes(self, origin: str, destination: str) -> RouteComparison:
        """
        Compare NORMAL shortest route with FLOOD-SAFE route.
        Computes additional distance, additional travel time, and risk reduction.
        """
        normal = self.find_normal_route(origin, destination)
        flood_safe = self.find_flood_safe_route(origin, destination)

        add_dist_m = round(flood_safe.distance_meters - normal.distance_meters, 1)
        add_dist_km = round(flood_safe.distance_km - normal.distance_km, 2)
        add_time_sec = round(flood_safe.estimated_travel_time_seconds - normal.estimated_travel_time_seconds, 1)
        add_time_min = round(flood_safe.estimated_travel_time_minutes - normal.estimated_travel_time_minutes, 1)
        risk_red = round(normal.mean_risk_score - flood_safe.mean_risk_score, 1)
        avoided_high_risk = normal.number_of_high_risk_segments - flood_safe.number_of_high_risk_segments

        # Recommend reroute if normal route has high-risk segments and safe route avoids them
        reroute = (normal.number_of_high_risk_segments > 0 and avoided_high_risk > 0) or (risk_red > 15.0)

        summary_lines = [
            "==================================================",
            "      FLOOD-AWARE ROAD ROUTE COMPARISON           ",
            "==================================================",
            f"Origin:       {origin}",
            f"Destination:  {destination}",
            "--------------------------------------------------",
            f"NORMAL ROUTE:     {normal.distance_km} km | {normal.estimated_travel_time_minutes} min | Mean Risk: {normal.mean_risk_score}/100 | High-Risk Segments: {normal.number_of_high_risk_segments}",
            f"FLOOD-SAFE ROUTE: {flood_safe.distance_km} km | {flood_safe.estimated_travel_time_minutes} min | Mean Risk: {flood_safe.mean_risk_score}/100 | High-Risk Segments: {flood_safe.number_of_high_risk_segments}",
            "--------------------------------------------------",
            f"Additional Distance:     {'+' if add_dist_km >= 0 else ''}{add_dist_km} km ({add_dist_m} m)",
            f"Additional Travel Time:  {'+' if add_time_min >= 0 else ''}{add_time_min} min ({add_time_sec} s)",
            f"Risk Score Reduction:    {risk_red} points (Max Risk: {normal.max_risk_score} -> {flood_safe.max_risk_score})",
            f"High-Risk Roads Avoided: {avoided_high_risk}",
            f"Recommendation:          {'REROUTE TO FLOOD-SAFE PATH' if reroute else 'NORMAL ROUTE SAFE'}",
            "==================================================",
        ]
        summary_str = "\n".join(summary_lines)

        return RouteComparison(
            normal_route=normal,
            flood_safe_route=flood_safe,
            additional_distance_meters=add_dist_m,
            additional_distance_km=add_dist_km,
            additional_travel_time_seconds=add_time_sec,
            additional_travel_time_minutes=add_time_min,
            risk_score_reduction=risk_red,
            high_risk_segments_avoided=avoided_high_risk,
            reroute_recommended=reroute,
            summary=summary_str,
        )

    def route_by_coordinates(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
    ) -> RouteComparison:
        """
        Calculate and compare routes between arbitrary (lat, lon) coordinates
        by snapping to the nearest road network nodes.
        """
        origin_node = self.graph_manager.get_nearest_node(origin_lat, origin_lon)
        dest_node = self.graph_manager.get_nearest_node(dest_lat, dest_lon)
        return self.compare_routes(origin_node, dest_node)

    def find_safe_route(
        self,
        origin_lat_lon: Tuple[float, float],
        dest_lat_lon: Tuple[float, float],
        vehicle_type: str = "car",
    ) -> Dict[str, Any]:
        """
        Compute optimal route penalizing flooded and impassable segments.
        Accepts lat-lon tuples.
        """
        comp = self.route_by_coordinates(
            origin_lat=origin_lat_lon[0],
            origin_lon=origin_lat_lon[1],
            dest_lat=dest_lat_lon[0],
            dest_lon=dest_lat_lon[1],
        )
        return asdict(comp.flood_safe_route)

    def find_k_alternative_routes(
        self,
        origin_lat_lon: Tuple[float, float],
        dest_lat_lon: Tuple[float, float],
        k: int = 3,
        vehicle_type: str = "car",
    ) -> List[Dict[str, Any]]:
        """
        Compute alternative route options.
        """
        comp = self.route_by_coordinates(
            origin_lat=origin_lat_lon[0],
            origin_lon=origin_lat_lon[1],
            dest_lat=dest_lat_lon[0],
            dest_lon=dest_lat_lon[1],
        )
        routes = [asdict(comp.flood_safe_route), asdict(comp.normal_route)]
        return routes[:k]
