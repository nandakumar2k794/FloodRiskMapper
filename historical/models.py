"""
Data models and schemas for historical flood and inundation data.
Enforces separation between OBSERVED, INFERRED, and SYNTHETIC evidence.
"""
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


class HistoricalEvidenceType(str, Enum):
    """
    Evidence classification for historical flooding:
    - OBSERVED: Directly recorded by satellite SAR (Sentinel-1), field surveys, or municipal GCC logs.
    - INFERRED: Derived from spatial buffer proximity, hydraulic continuity, or elevation depression matching.
    - SYNTHETIC: Derived from hydrodynamic / 2D hydraulic return-period simulations (e.g., 25-year flood line).
    """
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    SYNTHETIC = "SYNTHETIC"


class HistoricalFloodEvent(BaseModel):
    """Metadata for a documented major Chennai flood event."""
    event_id: str
    name: str
    year: int
    start_date: str
    end_date: str
    peak_24h_rainfall_mm: float
    description: str
    evidence_type: HistoricalEvidenceType = HistoricalEvidenceType.OBSERVED


class HistoricalFloodRecord(BaseModel):
    """Single spatial inundation record (polygon or point observation)."""
    record_id: str
    event_id: str
    evidence_type: HistoricalEvidenceType
    recorded_depth_cm: Optional[float] = None
    confidence_score: float = Field(1.0, ge=0.0, le=1.0)
    source_agency: str = "GCC"  # GCC, NRSC_ISRO, CMWSSB, COPERNICUS
    notes: Optional[str] = None


class HistoricalRoadFeatures(BaseModel):
    """
    Historical flood feature set for an individual road segment.
    Note: Lack of historical records is NOT proof of zero flood risk.
    """
    road_id: str
    historical_flood_count: int = Field(0, ge=0)
    historical_flood_frequency: float = Field(0.0, ge=0.0, le=1.0)
    historical_inundation_presence: bool = False
    observed_flood_count: int = Field(0, ge=0)
    inferred_flood_count: int = Field(0, ge=0)
    synthetic_flood_count: int = Field(0, ge=0)
    min_dist_to_flood_record_m: float = Field(float("inf"), ge=0.0)
    historical_confidence: float = Field(0.0, ge=0.0, le=1.0)
    historical_data_available: bool = False
    primary_evidence_type: Optional[HistoricalEvidenceType] = None
