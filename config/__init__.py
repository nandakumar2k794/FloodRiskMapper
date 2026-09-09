"""
Configuration package for Chennai Flood Access Risk Mapper with DrainWatch AI.
"""
from config.constants import (
    PROJECT_NAME,
    PROJECT_VERSION,
    CHENNAI_BBOX,
    CHENNAI_CENTER,
    CRS_WGS84,
    CRS_UTM44N,
    RISK_CLASSES,
    RISK_COLORS,
    RISK_WEIGHTS,
    ROUTE_RISK_ALPHA,
    DRAINWATCH_CLASSES,
)
from config.settings import settings
from config.logging_config import setup_logging, get_logger

__all__ = [
    "PROJECT_NAME",
    "PROJECT_VERSION",
    "CHENNAI_BBOX",
    "CHENNAI_CENTER",
    "CRS_WGS84",
    "CRS_UTM44N",
    "RISK_CLASSES",
    "RISK_COLORS",
    "RISK_WEIGHTS",
    "ROUTE_RISK_ALPHA",
    "DRAINWATCH_CLASSES",
    "settings",
    "setup_logging",
    "get_logger",
]
