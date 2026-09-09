"""
Pydantic request/response schemas for the API.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ─── Enums ─────────────────────────────────────────────────────────────────────
class RiskClass(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class VehicleType(str, Enum):
    CAR = "car"
    SUV = "suv"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    PEDESTRIAN = "pedestrian"


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DrainCondition(str, Enum):
    CLEAR = "drain_clear"
    BLOCKED_GARBAGE = "drain_blocked_garbage"
    BLOCKED_SILT = "drain_blocked_silt"
    OVERFLOWING = "drain_overflowing"
    STANDING_MINOR = "standing_water_minor"
    STANDING_MAJOR = "standing_water_major"
    ROAD_WATERLOGGED = "road_waterlogged"


# ─── Risk Map ──────────────────────────────────────────────────────────────────
from typing import List, Optional, Dict, Any, Union

class SegmentRisk(BaseModel):
    segment_id: Union[int, str]
    osm_id: Optional[int] = None
    name: Optional[str] = None
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_class: RiskClass
    accessible: bool
    predicted_depth_cm: Optional[float] = None
    rainfall_24h_mm: Optional[float] = None
    drainwatch_score: Optional[float] = None
    geometry: Dict[str, Any]  # GeoJSON LineString
    scored_at: datetime


class RiskMapResponse(BaseModel):
    type: str = "FeatureCollection"
    features: List[Dict[str, Any]]
    meta: Dict[str, Any]


# ─── Routing ───────────────────────────────────────────────────────────────────
class LatLon(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class RouteRequest(BaseModel):
    origin: LatLon
    destination: LatLon
    vehicle_type: VehicleType = VehicleType.CAR
    mode: str = "safest"   # "safest" | "fastest" | "avoid_flooded"


class RouteSegmentInfo(BaseModel):
    segment_id: Union[int, str]
    name: Optional[str]
    length_m: float
    risk_score: float
    risk_class: RiskClass
    accessible: bool


class Route(BaseModel):
    route_id: int
    geometry: Dict[str, Any]   # GeoJSON LineString
    total_length_m: float
    total_risk_score: float
    num_high_risk_segments: int
    num_inaccessible_segments: int
    estimated_time_min: float
    segments: List[RouteSegmentInfo]
    warnings: List[str]


class RouteResponse(BaseModel):
    routes: List[Route]
    recommended_route_id: int
    origin: LatLon
    destination: LatLon
    computed_at: datetime


# ─── Alerts ────────────────────────────────────────────────────────────────────
class Alert(BaseModel):
    alert_id: str
    segment_id: Optional[int] = None
    alert_type: str
    severity: AlertSeverity
    message: str
    location_name: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    created_at: datetime
    resolved_at: Optional[datetime] = None


class AlertsResponse(BaseModel):
    alerts: List[Alert]
    total: int
    critical_count: int
    high_count: int


# ─── DrainWatch ────────────────────────────────────────────────────────────────
class DrainObservation(BaseModel):
    camera_id: str
    location_name: Optional[str] = None
    lat: float
    lon: float
    road_segment_id: Optional[int] = None
    observed_at: datetime
    dominant_condition: DrainCondition
    blockage_score: float = Field(..., ge=0.0, le=1.0)
    overflow_score: float = Field(..., ge=0.0, le=1.0)
    waterlogging_score: float = Field(..., ge=0.0, le=1.0)
    composite_drain_score: float = Field(..., ge=0.0, le=1.0)
    frame_count: int
    thumbnail_url: Optional[str] = None


class DrainWatchResponse(BaseModel):
    observations: List[DrainObservation]
    total_cameras: int
    cameras_critical: int
    cameras_blocked: int
    last_updated: datetime


# ─── Rainfall ──────────────────────────────────────────────────────────────────
class RainfallReading(BaseModel):
    station_id: str
    station_name: str
    lat: float
    lon: float
    rainfall_mm: float
    recorded_at: datetime
    source: str


class RainfallSummary(BaseModel):
    stations: List[RainfallReading]
    city_avg_1h_mm: float
    city_avg_6h_mm: float
    city_avg_24h_mm: float
    imd_alert_level: str
    last_updated: datetime


# ─── Dashboard Stats ───────────────────────────────────────────────────────────
class DashboardStats(BaseModel):
    total_segments: int
    segments_critical: int
    segments_high: int
    segments_inaccessible: int
    rainfall_24h_avg_mm: float
    active_alerts: int
    drain_cameras_total: int
    drain_cameras_blocked: int
    last_updated: datetime
