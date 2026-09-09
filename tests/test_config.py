"""
Unit tests for configuration, constants, and logging setup.
"""
import pytest
import os
from pathlib import Path

from config.constants import (
    PROJECT_NAME,
    PROJECT_VERSION,
    CHENNAI_BBOX,
    CHENNAI_CENTER,
    CRS_WGS84,
    CRS_UTM44N,
    RISK_CLASSES,
    RISK_WEIGHTS,
    DEPTH_THRESHOLDS_CM,
    DRAINWATCH_CLASSES,
)
from config.settings import settings
from config.logging_config import setup_logging, get_logger, LOGS_DIR


class TestConstants:
    """Test validity of system constants."""

    def test_project_metadata(self):
        assert PROJECT_NAME == "Chennai Flood Access Risk Mapper with DrainWatch AI"
        assert PROJECT_VERSION.startswith("0.")

    def test_chennai_bounding_box(self):
        assert "north" in CHENNAI_BBOX
        assert "south" in CHENNAI_BBOX
        assert "east" in CHENNAI_BBOX
        assert "west" in CHENNAI_BBOX
        assert CHENNAI_BBOX["north"] > CHENNAI_BBOX["south"]
        assert CHENNAI_BBOX["east"] > CHENNAI_BBOX["west"]
        # Coordinates should be within Tamil Nadu / Chennai area
        assert 12.0 < CHENNAI_BBOX["south"] < 13.5
        assert 79.5 < CHENNAI_BBOX["west"] < 81.0

    def test_chennai_center(self):
        lat, lon = CHENNAI_CENTER
        assert CHENNAI_BBOX["south"] <= lat <= CHENNAI_BBOX["north"]
        assert CHENNAI_BBOX["west"] <= lon <= CHENNAI_BBOX["east"]

    def test_crs_definitions(self):
        assert CRS_WGS84 == "EPSG:4326"
        assert CRS_UTM44N == "EPSG:32644"

    def test_risk_weights_sum_to_one(self):
        total_weight = sum(RISK_WEIGHTS.values())
        assert abs(total_weight - 1.0) < 1e-6

    def test_risk_classes_ranges(self):
        assert set(RISK_CLASSES.keys()) == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert RISK_CLASSES["LOW"] == (0.00, 0.25)
        assert RISK_CLASSES["CRITICAL"] == (0.75, 1.00)

    def test_vehicle_depth_thresholds(self):
        assert "car" in DEPTH_THRESHOLDS_CM
        assert "suv" in DEPTH_THRESHOLDS_CM
        assert "bus" in DEPTH_THRESHOLDS_CM
        assert DEPTH_THRESHOLDS_CM["car"] < DEPTH_THRESHOLDS_CM["suv"] < DEPTH_THRESHOLDS_CM["bus"]

    def test_drainwatch_classes(self):
        assert "drain_clear" in DRAINWATCH_CLASSES
        assert "drain_blocked_garbage" in DRAINWATCH_CLASSES
        assert "road_waterlogged" in DRAINWATCH_CLASSES
        assert len(DRAINWATCH_CLASSES) >= 5


class TestSettings:
    """Test configuration settings object."""

    def test_settings_loaded(self):
        assert settings is not None
        assert settings.app_port == 8000
        assert settings.postgres_port == 5432
        assert "postgresql://" in settings.postgres_dsn_sync


class TestLogging:
    """Test logging setup."""

    def test_setup_logging(self):
        setup_logging(level="DEBUG", log_to_file=True)
        logger = get_logger("floodmap_test")
        logger.info("Test log entry")
        assert LOGS_DIR.exists()
