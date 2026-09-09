"""
Unit tests for Phase 1 placeholder interfaces.
Verifies modules define clean contracts without fake implementations.
"""
import pytest

from gis import DEMProcessor, HANDCalculator, RoadNetworkBuilder
from ml import RainfallService, FeatureExtractor, RoadFloodRiskModel
from cv import DrainDetector, DrainWatchService
from routing import RoadGraphManager, FloodRiskRouter


class TestGISPlaceholders:
    """Verify GIS placeholder contracts raise NotImplementedError."""

    def test_dem_processor_interface(self):
        proc = DEMProcessor()
        with pytest.raises(NotImplementedError):
            proc.load_dem("dummy.tif")
        with pytest.raises(NotImplementedError):
            proc.compute_slope()
        with pytest.raises(NotImplementedError):
            proc.get_elevation_at(13.0827, 80.2707)

    def test_hand_calculator_interface(self):
        calc = HANDCalculator()
        with pytest.raises(NotImplementedError):
            calc.calculate_hand("dem.tif", "drainage.shp", "out.tif")
        with pytest.raises(NotImplementedError):
            calc.get_hand_value(13.0827, 80.2707)

    def test_road_network_builder_interface(self):
        builder = RoadNetworkBuilder()
        with pytest.raises(NotImplementedError):
            builder.fetch_network_from_osm({"north": 13.1, "south": 13.0, "east": 80.2, "west": 80.1})
        with pytest.raises(NotImplementedError):
            builder.enrich_segments("dem.tif", "hand.tif")


class TestMLPlaceholders:
    """Verify ML and rainfall placeholder contracts."""

    def test_rainfall_service_interface(self):
        service = RainfallService()
        with pytest.raises(NotImplementedError):
            service.fetch_live_rainfall(13.0827, 80.2707)
        with pytest.raises(NotImplementedError):
            service.get_accumulation(13.0827, 80.2707, 24)

    def test_feature_extractor_interface(self):
        extractor = FeatureExtractor()
        with pytest.raises(NotImplementedError):
            extractor.extract_features("seg_123", 45.0, 1.2, 0.05, 0.8, 0.9)

    def test_flood_risk_model_interface(self):
        model = RoadFloodRiskModel()
        with pytest.raises(NotImplementedError):
            model.load_model("model.pkl")
        with pytest.raises(NotImplementedError):
            model.predict_risk({"rainfall": 50.0})


class TestCVInterfaces:
    """Verify DrainWatch CV contracts and handling."""

    def test_detector_interface(self):
        detector = DrainDetector()
        assert detector.conf_threshold == 0.25
        with pytest.raises(FileNotFoundError):
            detector.load_weights("non_existent_weights.pt")

    def test_drainwatch_service_interface(self):
        service = DrainWatchService()
        cond = service.get_latest_condition("cam_01")
        assert cond.is_valid is False
        assert cond.camera_id == "cam_01"


class TestRoutingInterfaces:
    """Verify routing system contracts and empty graph handling."""

    def test_graph_manager_interface(self):
        mgr = RoadGraphManager()
        G = mgr.build_from_geodataframe(None)
        assert len(G.nodes) == 0
        mgr.update_edge_risk_weights({})

    def test_router_interface(self):
        mgr = RoadGraphManager.create_chennai_arterial_network()
        router = FloodRiskRouter(graph_manager=mgr)
        route = router.find_safe_route((13.0827, 80.2755), (13.0418, 80.2341))
        assert "distance_km" in route
        assert "estimated_travel_time_minutes" in route

        alts = router.find_k_alternative_routes((13.0827, 80.2755), (13.0418, 80.2341), k=2)
        assert len(alts) <= 2
