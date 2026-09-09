"""
Data models and enumeration types for Chennai rainfall pipeline.
Explicitly distinguishes REAL observed rainfall from SIMULATED scenario rainfall.
"""
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class RainfallSourceType(str, Enum):
    """
    Source distinction:
    - REAL: Observations from physical automated weather stations (AWS), IMD gauges, or live APIs.
    - SIMULATED: Generated from what-if scenarios, return-period design storms, or synthetic hyetographs.
    """
    REAL = "REAL"
    SIMULATED = "SIMULATED"


class RainfallStation(BaseModel):
    """Metadata for an automated weather station / rain gauge."""
    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation_m: Optional[float] = None
    is_active: bool = True
    source_agency: str = "IMD"  # e.g., IMD, CMWSSB, GCC, OpenWeather


class RainfallReading(BaseModel):
    """
    Single discrete rainfall measurement at a point in time.
    """
    station_id: str
    timestamp: datetime
    precipitation_mm: float = Field(..., ge=0.0)
    source_type: RainfallSourceType
    quality_flag: str = "VALID"  # VALID, IMPUTED, SUSPECT, INTERPOLATED

    @field_validator("precipitation_mm")
    @classmethod
    def validate_precipitation(cls, v: float) -> float:
        # Max theoretical 15-min rainfall rate ever recorded ~100mm/15min
        if v < 0:
            raise ValueError("Precipitation cannot be negative")
        return round(float(v), 2)


class RainfallAggregates(BaseModel):
    """
    Standardized multi-interval accumulated rainfall windows required by the risk model.
    """
    rainfall_15min: float = Field(0.0, ge=0.0)
    rainfall_1h: float = Field(0.0, ge=0.0)
    rainfall_3h: float = Field(0.0, ge=0.0)
    rainfall_6h: float = Field(0.0, ge=0.0)
    rainfall_24h: float = Field(0.0, ge=0.0)
    source_type: RainfallSourceType = RainfallSourceType.REAL
    as_of_timestamp: Optional[datetime] = None


class ScenarioParameters(BaseModel):
    """
    Definition of a scenario rainfall event (what-if or design storm).
    """
    scenario_id: str
    name: str
    total_24h_mm: float = Field(..., ge=0.0)
    distribution_type: str = "CONSTANT"  # CONSTANT, HUFF_II, PEAK_FRONT, PEAK_CENTER
    spatial_extent: str = "CHENNAI_UNIFORM"  # CHENNAI_UNIFORM, NORTH_HOTSPOT, SOUTH_HOTSPOT
    source_type: RainfallSourceType = RainfallSourceType.SIMULATED
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)
