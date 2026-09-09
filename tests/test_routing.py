"""
Unit tests for Flood-Aware Road Routing System.
Tests:
- RoadGraphManager graph construction from GeoDataFrame
- Travel time and flood risk impedance cost computation
- Dynamic edge risk weight updates
- Normal shortest path vs. Flood-safe path detour behavior
- Both Dijkstra and A* pathfinding algorithms
- Extraction of all required route metrics:
    * distance
    * estimated travel time
    * risk score (mean and max)
    * number of high-risk segments
    * additional distance
    * additional travel time
- Route comparison between NORMAL and FLOOD-SAFE routes
- Routing on real Chennai arterial network using geographic coordinates
"""
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point

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


@pytest.fixture
def triangle_network_gdf():
    """
    Construct a simple 3-node road network: A -> B -> C and direct detour A -> C.
    A: (13.08, 80.27)
    B: (13.06, 80.25)
    C: (13.04, 80.23)
    Direct path: A -> B -> C (total 6 km, fast primary road)
    Detour path: A -> C (longer 9 km secondary road)
    """
    edges = [
        {
            "u": "node_A",
            "v": "node_B",
            "segment_id": "seg_AB",
            "name": "Arterial Road North",
            "highway": "primary",
            "length_m": 3000.0,
            "maxspeed": 60.0,
            "flood_access_risk_score": 0.0,
            "geometry": LineString([(80.27, 13.08), (80.25, 13.06)]),
        },
        {
            "u": "node_B",
            "v": "node_C",
            "segment_id": "seg_BC",
            "name": "Arterial Road South",
            "highway": "primary",
            "length_m": 3000.0,
            "maxspeed": 60.0,
            "flood_access_risk_score": 0.0,
            "geometry": LineString([(80.25, 13.06), (80.23, 13.04)]),
        },
        {
            "u": "node_A",
            "v": "node_C",
            "segment_id": "seg_AC",
            "name": "Bypass Elevated Way",
            "highway": "secondary",
            "length_m": 8500.0,
            "maxspeed": 50.0,
            "flood_access_risk_score": 5.0,
            "geometry": LineString([(80.27, 13.08), (80.23, 13.04)]),
        },
    ]
    return gpd.GeoDataFrame(edges, crs="EPSG:4326")


class TestRoadGraphManager:
    """Tests for RoadGraphManager."""

    def test_build_from_geodataframe(self, triangle_network_gdf):
        manager = RoadGraphManager()
        G = manager.build_from_geodataframe(triangle_network_gdf)

        assert len(G.nodes) == 3
        assert G.has_edge("node_A", "node_B")
        assert G.has_edge("node_B", "node_C")
        assert G.has_edge("node_A", "node_C")

        edge_ab = G["node_A"]["node_B"]
        assert edge_ab["length"] == 3000.0
        assert edge_ab["speed_kph"] == 60.0
        assert edge_ab["travel_time"] == pytest.approx(180.0)  # 3000m / (60kmh/3.6 = 16.67m/s) = 180s
        assert edge_ab["risk_score"] == 0.0

    def test_dynamic_risk_weight_update(self, triangle_network_gdf):
        manager = RoadGraphManager()
        manager.build_from_geodataframe(triangle_network_gdf)

        # Update seg_AB to CRITICAL risk (85.0)
        manager.update_edge_risk_weights({"seg_AB": 85.0})

        edge_ab = manager.graph["node_A"]["node_B"]
        assert edge_ab["risk_score"] == 85.0
        assert edge_ab["risk_class"] == "CRITICAL"
        # Flood cost should be severely penalized
        assert edge_ab["flood_cost"] > 1000.0

    def test_nearest_node_lookup(self, triangle_network_gdf):
        manager = RoadGraphManager()
        manager.build_from_geodataframe(triangle_network_gdf)

        # Query point very close to node_A (13.08, 80.27)
        nearest = manager.get_nearest_node(13.081, 80.271)
        assert nearest == "node_A"

        # Query point very close to node_C (13.04, 80.23)
        nearest_c = manager.get_nearest_node(13.039, 80.231)
        assert nearest_c == "node_C"

    def test_real_chennai_arterial_network(self):
        manager = RoadGraphManager.create_chennai_arterial_network()
        assert len(manager.graph.nodes) >= 10
        assert len(manager.graph.edges) >= 20

        # Check key Chennai landmarks are in graph
        for landmark in ["Central", "Adyar", "Guindy", "Velachery", "T_Nagar", "Sholinganallur"]:
            assert landmark in manager.graph.nodes


class TestFloodAwareRouter:
    """Tests for FloodAwareRouter routing and detour behavior."""

    def test_normal_route_selects_fastest_path(self, triangle_network_gdf):
        manager = RoadGraphManager()
        manager.build_from_geodataframe(triangle_network_gdf)
        router = FloodAwareRouter(manager, algorithm="dijkstra")

        # Without flood risk, A -> B -> C is faster (360s) than A -> C (612s)
        normal_route = router.find_normal_route("node_A", "node_C")
        assert normal_route.nodes == ["node_A", "node_B", "node_C"]
        assert normal_route.distance_meters == 6000.0
        assert normal_route.estimated_travel_time_seconds == pytest.approx(360.0)

    def test_flood_safe_route_detours_around_flooded_segment(self, triangle_network_gdf):
        manager = RoadGraphManager()
        manager.build_from_geodataframe(triangle_network_gdf)
        router = FloodAwareRouter(manager, algorithm="dijkstra")

        # Inundate middle segment seg_BC with 90% risk
        manager.update_edge_risk_weights({"seg_BC": 90.0})

        # Normal route still chooses A -> B -> C because it only cares about travel time
        normal_route = router.find_normal_route("node_A", "node_C")
        assert normal_route.nodes == ["node_A", "node_B", "node_C"]
        assert normal_route.number_of_high_risk_segments == 1

        # Flood-safe route DETOURS along A -> C to avoid the flooded segment
        safe_route = router.find_flood_safe_route("node_A", "node_C")
        assert safe_route.nodes == ["node_A", "node_C"]
        assert safe_route.number_of_high_risk_segments == 0
        assert safe_route.max_risk_score < 50.0

    def test_astar_matches_dijkstra(self, triangle_network_gdf):
        manager = RoadGraphManager()
        manager.build_from_geodataframe(triangle_network_gdf)

        router_dijkstra = FloodAwareRouter(manager, algorithm="dijkstra")
        router_astar = FloodAwareRouter(manager, algorithm="astar")

        norm_d = router_dijkstra.find_normal_route("node_A", "node_C")
        norm_a = router_astar.find_normal_route("node_A", "node_C")

        assert norm_d.nodes == norm_a.nodes
        assert norm_d.distance_meters == norm_a.distance_meters

    def test_compare_routes_output_structure(self, triangle_network_gdf):
        manager = RoadGraphManager()
        manager.build_from_geodataframe(triangle_network_gdf)
        manager.update_edge_risk_weights({"seg_BC": 95.0})

        router = FloodAwareRouter(manager)
        comp = router.compare_routes("node_A", "node_C")

        assert isinstance(comp, RouteComparison)
        assert comp.normal_route.route_type == "NORMAL"
        assert comp.flood_safe_route.route_type == "FLOOD_SAFE"

        # Additional distance should be positive (detour is longer)
        assert comp.additional_distance_meters > 0
        assert comp.additional_distance_km > 0

        # Additional travel time should be positive
        assert comp.additional_travel_time_seconds > 0

        # Risk score should be reduced
        assert comp.risk_score_reduction > 0
        assert comp.high_risk_segments_avoided == 1
        assert comp.reroute_recommended is True

        # Verify summary string
        assert "FLOOD-AWARE ROAD ROUTE COMPARISON" in comp.summary
        assert "NORMAL ROUTE:" in comp.summary
        assert "FLOOD-SAFE ROUTE:" in comp.summary
        assert "REROUTE TO FLOOD-SAFE PATH" in comp.summary

    def test_real_chennai_network_routing_and_comparison(self):
        """
        Test realistic emergency route from Central to Sholinganallur.
        Simulates heavy flooding on Velachery Main Road.
        """
        manager = RoadGraphManager.create_chennai_arterial_network()

        # Simulate heavy waterlogging on Velachery routes
        manager.update_edge_risk_weights({
            "Guindy_Velachery": 95.0,
            "Saidapet_Velachery": 90.0,
            "Velachery_Sholinganallur": 85.0,
        })

        router = FloodAwareRouter(manager, algorithm="dijkstra")

        # Route from Central (13.0827, 80.2755) to Sholinganallur (12.9010, 80.2279)
        comp = router.route_by_coordinates(
            origin_lat=13.0827, origin_lon=80.2755,
            dest_lat=12.9010, dest_lon=80.2279,
        )

        assert comp.normal_route.distance_km > 10.0
        assert comp.flood_safe_route.distance_km > 10.0

        # Flood safe route should avoid Velachery flooded links
        assert "Velachery" not in comp.flood_safe_route.nodes
        assert comp.flood_safe_route.number_of_high_risk_segments == 0
