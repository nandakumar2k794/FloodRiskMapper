"""
Road network graph representation and dynamic risk weight assignment for Chennai.
Builds NetworkX DiGraphs with travel time and flood-risk impedance costs.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString

from config.constants import (
    CRS_WGS84,
    CRS_UTM44N,
    ROUTE_RISK_ALPHA,
    ROUTE_INACCESSIBLE_PENALTY,
    DEPTH_THRESHOLDS_CM,
)


# Default free-flow speed limits in km/h by OpenStreetMap highway type
DEFAULT_SPEED_LIMITS_KPH: Dict[str, float] = {
    "motorway": 80.0,
    "trunk": 70.0,
    "primary": 50.0,
    "secondary": 40.0,
    "tertiary": 30.0,
    "residential": 25.0,
    "unclassified": 25.0,
    "living_street": 15.0,
    "service": 15.0,
}
DEFAULT_SPEED_KPH = 30.0


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance in meters between two lat-lon points."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def compute_flood_cost(
    travel_time_sec: float,
    risk_score: float,
    alpha: float = ROUTE_RISK_ALPHA,
    inaccessible_threshold: float = 75.0,
    inaccessible_penalty: float = ROUTE_INACCESSIBLE_PENALTY,
) -> float:
    """
    Calculate dynamic flood-aware traversal cost.

    Formula:
        cost = travel_time * (1.0 + alpha * (risk_score / 100.0)^2) + penalty
    High-risk roads (risk_score >= 75) incur a heavy penalty to strongly encourage detour.
    """
    r_norm = min(1.0, max(0.0, risk_score / 100.0))
    # Quadratic risk penalty: low risk adds negligible delay, high risk exponentially inflates cost
    multiplier = 1.0 + alpha * (r_norm ** 2)
    base_cost = travel_time_sec * multiplier

    # Extra severe penalty if segment is critical / inaccessible
    penalty = inaccessible_penalty if risk_score >= inaccessible_threshold else 0.0
    return float(base_cost + penalty)


class RoadGraphManager:
    """
    Manages spatial road network graphs with dynamic flood risk weights.
    Supports both normal travel-time routing and flood-aware penalty routing.
    """

    def __init__(self, alpha: float = ROUTE_RISK_ALPHA) -> None:
        self.alpha = alpha
        self.graph: nx.DiGraph = nx.DiGraph()
        self.node_coords: Dict[Any, Tuple[float, float]] = {}  # node -> (lat, lon)
        self._segment_to_edges: Dict[str, List[Tuple[Any, Any]]] = {}

    def build_from_geodataframe(
        self,
        road_edges_gdf: gpd.GeoDataFrame,
        risk_column: Optional[str] = "flood_access_risk_score",
    ) -> nx.DiGraph:
        """
        Construct a NetworkX DiGraph from spatial road edges GeoDataFrame.
        Extracts coordinates, lengths, travel times, and initial risk weights.
        """
        if road_edges_gdf is None or road_edges_gdf.empty:
            self.graph = nx.DiGraph()
            return self.graph

        G = nx.DiGraph()
        self.node_coords.clear()
        self._segment_to_edges.clear()

        for idx, row in road_edges_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            # Determine start and end coordinates
            if isinstance(geom, LineString):
                coords = list(geom.coords)
                start_pt = coords[0]
                end_pt = coords[-1]
            else:
                bounds = geom.bounds  # (minx, miny, maxx, maxy)
                start_pt = (bounds[0], bounds[1])
                end_pt = (bounds[2], bounds[3])

            # Node IDs based on coordinates or explicit u, v
            u = str(row.get("u", f"node_{start_pt[0]:.5f}_{start_pt[1]:.5f}"))
            v = str(row.get("v", f"node_{end_pt[0]:.5f}_{end_pt[1]:.5f}"))

            # Store geographic coordinates: start_pt is (lon, lat)
            self.node_coords[u] = (float(start_pt[1]), float(start_pt[0]))
            self.node_coords[v] = (float(end_pt[1]), float(end_pt[0]))

            # Edge attributes
            length_m = float(row.get("length_m", row.get("length", 0.0)))
            if length_m <= 0:
                length_m = haversine_distance_m(
                    self.node_coords[u][0], self.node_coords[u][1],
                    self.node_coords[v][0], self.node_coords[v][1],
                )
                length_m = max(10.0, length_m)

            hw_type = str(row.get("highway", "residential")).lower()
            speed_kph = float(row.get("maxspeed", DEFAULT_SPEED_LIMITS_KPH.get(hw_type, DEFAULT_SPEED_KPH)))
            speed_mps = max(2.0, (speed_kph * 1000.0) / 3600.0)

            travel_time_sec = float(length_m / speed_mps)

            # Risk score (0 - 100)
            risk_val = float(row.get(risk_column, 0.0)) if risk_column and risk_column in row else 0.0

            flood_cost = compute_flood_cost(travel_time_sec, risk_val, alpha=self.alpha)

            seg_id = str(row.get("segment_id", row.get("road_id", f"seg_{idx}")))

            edge_data = {
                "segment_id": seg_id,
                "length": length_m,
                "speed_kph": speed_kph,
                "travel_time": travel_time_sec,
                "risk_score": risk_val,
                "risk_class": "CRITICAL" if risk_val >= 75 else ("HIGH" if risk_val >= 50 else ("MODERATE" if risk_val >= 25 else "LOW")),
                "flood_cost": flood_cost,
                "highway": hw_type,
                "geometry": geom,
            }

            # Forward edge
            G.add_edge(u, v, **edge_data)
            self._segment_to_edges.setdefault(seg_id, []).append((u, v))

            # Bidirectional unless oneway=True
            is_oneway = bool(row.get("oneway", False))
            if not is_oneway:
                rev_data = edge_data.copy()
                G.add_edge(v, u, **rev_data)
                self._segment_to_edges[seg_id].append((v, u))

        # Assign node coordinates as attributes in NetworkX
        for node, (lat, lon) in self.node_coords.items():
            if node in G:
                G.nodes[node]["y"] = lat
                G.nodes[node]["x"] = lon

        self.graph = G
        return self.graph

    def update_edge_risk_weights(
        self,
        segment_risks: Dict[str, float],
        alpha: Optional[float] = None,
        inaccessible_threshold: float = 75.0,
    ) -> None:
        """
        Dynamically update risk scores and flood costs on all matching edges in the graph.

        Parameters:
            segment_risks: Map of segment_id -> risk_score (0.0 to 100.0).
            alpha: Risk amplification multiplier (defaults to self.alpha).
            inaccessible_threshold: Score at which severe penalty is triggered.
        """
        active_alpha = alpha if alpha is not None else self.alpha

        for seg_id, risk_score in segment_risks.items():
            edges = self._segment_to_edges.get(seg_id, [])
            for u, v in edges:
                if self.graph.has_edge(u, v):
                    data = self.graph[u][v]
                    data["risk_score"] = float(risk_score)
                    data["risk_class"] = (
                        "CRITICAL" if risk_score >= 75.0
                        else ("HIGH" if risk_score >= 50.0
                              else ("MODERATE" if risk_score >= 25.0 else "LOW"))
                    )
                    data["flood_cost"] = compute_flood_cost(
                        travel_time_sec=data["travel_time"],
                        risk_score=float(risk_score),
                        alpha=active_alpha,
                        inaccessible_threshold=inaccessible_threshold,
                    )

    def get_nearest_node(self, lat: float, lon: float) -> str:
        """
        Find the nearest graph node to a given (lat, lon) query point.
        """
        if not self.node_coords:
            raise ValueError("Graph has no nodes with spatial coordinates.")

        best_node = None
        min_dist = float("inf")

        for node, (n_lat, n_lon) in self.node_coords.items():
            dist = haversine_distance_m(lat, lon, n_lat, n_lon)
            if dist < min_dist:
                min_dist = dist
                best_node = str(node)

        return best_node

    @classmethod
    def create_chennai_arterial_network(cls) -> "RoadGraphManager":
        """
        Construct a real Chennai arterial road network graph connecting key junctions
        with real geographic coordinates (WGS84 EPSG:4326).
        Junctions:
          - Chennai Central / Ripon Building: (13.0827, 80.2755)
          - Anna Salai / Thousand Lights: (13.0560, 80.2520)
          - T. Nagar / Panagal Park: (13.0418, 80.2341)
          - Saidapet / Maraimalai Adigal Bridge: (13.0210, 80.2230)
          - Guindy / Kathipara Junction: (13.0067, 80.2025)
          - Adyar / Malar Hospital: (13.0067, 80.2558)
          - Velachery / Vijaya Nagar (Flood Hotspot): (12.9815, 80.2180)
          - OMR / SRP Tools / Taramani: (12.9850, 80.2450)
          - Sholinganallur Junction: (12.9010, 80.2279)
          - Anna Nagar Roundtana: (13.0850, 80.2101)
          - Koyambedu / CMBT: (13.0690, 80.1940)
          - Marina Beach / Kamarajar Salai: (13.0500, 80.2820)
        """
        junctions = {
            "Central": (13.0827, 80.2755),
            "Anna_Nagar": (13.0850, 80.2101),
            "Koyambedu": (13.0690, 80.1940),
            "Thousand_Lights": (13.0560, 80.2520),
            "Marina": (13.0500, 80.2820),
            "T_Nagar": (13.0418, 80.2341),
            "Saidapet": (13.0210, 80.2230),
            "Guindy": (13.0067, 80.2025),
            "Adyar": (13.0067, 80.2558),
            "Velachery": (12.9815, 80.2180),
            "Taramani": (12.9850, 80.2450),
            "Sholinganallur": (12.9010, 80.2279),
        }

        # Real arterial links (connecting edges)
        arterial_links = [
            ("Central", "Thousand_Lights", "Anna Salai (North)", "primary", 3800.0, 50.0),
            ("Central", "Marina", "Kamarajar Salai", "primary", 3600.0, 50.0),
            ("Central", "Anna_Nagar", "Poonamallee High Road", "primary", 7200.0, 50.0),
            ("Anna_Nagar", "Koyambedu", "Jawaharlal Nehru Road", "primary", 2400.0, 50.0),
            ("Koyambedu", "Guindy", "Inner Ring Road", "primary", 7500.0, 55.0),
            ("Thousand_Lights", "T_Nagar", "G.N. Chetty Road", "secondary", 2200.0, 40.0),
            ("Thousand_Lights", "Marina", "Dr. Radhakrishnan Salai", "secondary", 3200.0, 40.0),
            ("Marina", "Adyar", "Santhome High Road", "primary", 5200.0, 45.0),
            ("T_Nagar", "Saidapet", "Anna Salai (Central)", "primary", 2600.0, 50.0),
            ("Saidapet", "Guindy", "Anna Salai (South)", "trunk", 2600.0, 60.0),
            ("Saidapet", "Adyar", "Sardar Patel Road (West)", "primary", 3800.0, 45.0),
            ("Guindy", "Adyar", "Sardar Patel Road (Main)", "primary", 6100.0, 50.0),
            ("Guindy", "Velachery", "Velachery Main Road (North)", "secondary", 3400.0, 35.0),
            ("Saidapet", "Velachery", "Adambakkam Link", "secondary", 4200.0, 35.0),
            ("Velachery", "Taramani", "Taramani Link Road", "secondary", 3100.0, 40.0),
            ("Adyar", "Taramani", "Rajiv Gandhi Salai (OMR North)", "trunk", 2800.0, 60.0),
            ("Taramani", "Sholinganallur", "OMR IT Expressway", "trunk", 9600.0, 70.0),
            ("Velachery", "Sholinganallur", "Pallikaranai Radial Road", "secondary", 8800.0, 45.0),
        ]

        edges_data = []
        for u, v, name, hw, length, speed in arterial_links:
            p1 = junctions[u]
            p2 = junctions[v]
            geom = LineString([(p1[1], p1[0]), (p2[1], p2[0])])  # (lon, lat)
            edges_data.append({
                "u": u,
                "v": v,
                "name": name,
                "segment_id": f"{u}_{v}",
                "highway": hw,
                "length_m": length,
                "maxspeed": speed,
                "oneway": False,
                "flood_access_risk_score": 0.0,
                "geometry": geom,
            })

        edges_gdf = gpd.GeoDataFrame(edges_data, crs=CRS_WGS84)
        manager = cls()
        manager.build_from_geodataframe(edges_gdf)
        return manager
