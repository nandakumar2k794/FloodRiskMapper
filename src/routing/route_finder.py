"""
Route Finder — risk-weighted shortest paths using NetworkX.
Implements Yen's K-shortest paths algorithm with flood-aware edge weights.
"""
import logging
import math
import heapq
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any, Optional

import networkx as nx

from config.constants import (
    ROUTE_RISK_ALPHA, ROUTE_INACCESSIBLE_PENALTY,
    ROUTE_K_ALTERNATIVES, DEPTH_THRESHOLDS_CM,
)
from src.ml.predictor import RiskPredictor
from src.api.schemas import Route, RouteSegmentInfo, RiskClass

logger = logging.getLogger(__name__)


def _haversine(lat1, lon1, lat2, lon2) -> float:
    """Distance in metres between two lat/lon points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def _edge_weight(length_m: float, risk_score: float, accessible: bool, alpha: float) -> float:
    penalty = ROUTE_INACCESSIBLE_PENALTY if not accessible else 1.0
    return length_m * (1 + alpha * risk_score) * penalty


class RouteFinder:
    """
    Builds a risk-weighted directed graph from scored road segments
    and finds K-shortest safe routes using Yen's algorithm.
    """

    def __init__(self, _predictor=None):
        # Accept an injected predictor (shared singleton) to avoid loading
        # the XGBoost model twice when app_state pre-warms at startup.
        self._predictor = _predictor if _predictor is not None else RiskPredictor()
        self._graph_cache: Dict[str, nx.DiGraph] = {}
        self._segment_map: Dict[int, Dict] = {}

    async def _build_graph(self, vehicle_type: str) -> nx.DiGraph:
        # Return cached graph for this vehicle type
        if vehicle_type in self._graph_cache:
            return self._graph_cache[vehicle_type]

        depth_limit = DEPTH_THRESHOLDS_CM.get(vehicle_type, 25)
        segments = await self._predictor.get_scored_segments()

        G = nx.DiGraph()
        self._segment_map = {}
        all_nodes = []  # [(lat, lon)] to link into connected graph

        for seg in segments:
            coords = seg["geometry"]["coordinates"]
            if len(coords) < 2:
                continue

            start = coords[0]   # [lon, lat]
            end = coords[-1]
            u = (round(start[1], 5), round(start[0], 5))  # (lat, lon)
            v = (round(end[1],   5), round(end[0],   5))

            risk = seg["risk_score"]
            depth = seg.get("predicted_depth_cm", 0.0) or 0.0
            accessible = depth < depth_limit and risk < 0.9
            length = seg.get("length_m", 100.0)
            w = _edge_weight(length, risk, accessible, ROUTE_RISK_ALPHA)

            G.add_node(u, lat=u[0], lon=u[1])
            G.add_node(v, lat=v[0], lon=v[1])
            G.add_edge(u, v,
                weight=w, length_m=length, risk_score=risk,
                accessible=accessible, segment_id=seg["segment_id"],
                name=seg.get("name", ""), risk_class=seg["risk_class"],
            )
            G.add_edge(v, u,
                weight=w, length_m=length, risk_score=risk,
                accessible=accessible, segment_id=seg["segment_id"],
                name=seg.get("name", ""), risk_class=seg["risk_class"],
            )
            all_nodes.extend([u, v])

        self._segment_map = {seg["segment_id"]: seg for seg in segments}

        # ── Connect isolated nodes into a single component ─────────────────────
        # For each node, find its K nearest neighbours and add connector edges.
        # This ensures shortest_simple_paths can always find a path across Chennai.
        K_NEIGHBOURS = 3
        MAX_LINK_M = 4000  # only link nodes within 4 km
        nodes_list = list(G.nodes)
        for node in nodes_list:
            distances = []
            for other in nodes_list:
                if other == node:
                    continue
                d = _haversine(node[0], node[1], other[0], other[1])
                distances.append((d, other))
            distances.sort()
            for d, neighbour in distances[:K_NEIGHBOURS]:
                if d > MAX_LINK_M:
                    break
                if not G.has_edge(node, neighbour):
                    # Connector edge: low base risk (it's a virtual road link)
                    base_risk = 0.1
                    w_conn = _edge_weight(d, base_risk, True, ROUTE_RISK_ALPHA)
                    G.add_edge(node, neighbour,
                        weight=w_conn, length_m=round(d, 1), risk_score=base_risk,
                        accessible=True, segment_id=-1,
                        name="Connector", risk_class="LOW",
                    )
                    G.add_edge(neighbour, node,
                        weight=w_conn, length_m=round(d, 1), risk_score=base_risk,
                        accessible=True, segment_id=-1,
                        name="Connector", risk_class="LOW",
                    )

        logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (vehicle={vehicle_type})")
        self._graph_cache[vehicle_type] = G
        return G

    def _nearest_node(self, G: nx.DiGraph, lat: float, lon: float):
        """Find graph node nearest to a lat/lon coordinate."""
        best, best_d = None, float("inf")
        for node in G.nodes:
            d = _haversine(lat, lon, node[0], node[1])
            if d < best_d:
                best, best_d = node, d
        return best, best_d

    async def find(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        vehicle_type: str = "car",
        mode: str = "safest",
    ) -> List[Route]:
        G = await self._build_graph(vehicle_type)

        src, src_d = self._nearest_node(G, *origin)
        dst, dst_d = self._nearest_node(G, *destination)

        if src is None or dst is None or src == dst:
            return []

        logger.info(f"Routing {src} → {dst} (mode={mode}, vehicle={vehicle_type})")

        # Use simple_paths fallback if yen's fails
        try:
            k_paths = list(nx.shortest_simple_paths(G, src, dst, weight="weight"))
            k_paths = k_paths[:ROUTE_K_ALTERNATIVES]
        except nx.NetworkXNoPath:
            logger.warning("No path found")
            return []
        except Exception as e:
            logger.exception(f"Routing error: {e}")
            return []

        routes = []
        for idx, path in enumerate(k_paths):
            route = self._build_route(G, path, idx + 1, vehicle_type)
            routes.append(route)

        # Sort by mode preference
        if mode == "safest":
            routes.sort(key=lambda r: (r.total_risk_score, r.total_length_m))
        elif mode == "fastest":
            routes.sort(key=lambda r: r.total_length_m)
        else:  # avoid_flooded
            routes.sort(key=lambda r: (r.num_inaccessible_segments, r.total_risk_score))

        return routes

    def _build_route(self, G: nx.DiGraph, path: list, route_id: int, vehicle_type: str) -> Route:
        coords = []
        segments_info = []
        total_length = 0.0
        total_risk = 0.0
        warnings = []

        for u, v in zip(path[:-1], path[1:]):
            data = G.edges[u, v]
            coords.append([u[1], u[0]])  # [lon, lat]

            total_length += data["length_m"]
            total_risk += data["risk_score"] * data["length_m"]

            seg_info = RouteSegmentInfo(
                segment_id=data["segment_id"],
                name=data["name"] or "Unnamed Road",
                length_m=round(data["length_m"], 1),
                risk_score=round(data["risk_score"], 3),
                risk_class=RiskClass(data["risk_class"]),
                accessible=data["accessible"],
            )
            segments_info.append(seg_info)

            if data["risk_class"] == "CRITICAL":
                warnings.append(f"⚠️ {data['name'] or 'Segment'} is CRITICAL risk")
            if not data["accessible"]:
                warnings.append(f"🚫 {data['name'] or 'Segment'} may be inaccessible")

        if path:
            last = path[-1]
            coords.append([last[1], last[0]])

        avg_risk = total_risk / max(total_length, 1)
        high_risk_count = sum(1 for s in segments_info if s.risk_class in ["HIGH", "CRITICAL"])
        inaccess_count = sum(1 for s in segments_info if not s.accessible)

        # Remove duplicate warnings
        warnings = list(dict.fromkeys(warnings))

        return Route(
            route_id=route_id,
            geometry={"type": "LineString", "coordinates": coords},
            total_length_m=round(total_length, 1),
            total_risk_score=round(avg_risk, 3),
            num_high_risk_segments=high_risk_count,
            num_inaccessible_segments=inaccess_count,
            estimated_time_min=round(total_length / 1000 / 30 * 60, 1),  # 30 km/h avg
            segments=segments_info,
            warnings=warnings[:5],  # cap warnings
        )
