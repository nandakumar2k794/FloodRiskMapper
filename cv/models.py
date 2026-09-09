"""
Data models and enumeration types for DrainWatch AI computer vision pipeline.
Target classes:
- drain_inlet, open_drain, covered_drain
- garbage, plastic_waste, leaves, silt_mud, construction_debris
- blocked_drain
- standing_water, flowing_water, overflowing_drain, road_waterlogging

Water depth in cm is NEVER claimed without calibrated camera geometry.
Water-rise rate is expressed as Δ(water_coverage_%) per minute — a relative proxy,
not an absolute depth measurement.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import numpy as np


class DrainWatchClass(str, Enum):
    """Target object detection and classification categories."""
    DRAIN_INLET = "drain_inlet"
    OPEN_DRAIN = "open_drain"
    COVERED_DRAIN = "covered_drain"
    GARBAGE = "garbage"
    PLASTIC_WASTE = "plastic_waste"
    LEAVES = "leaves"
    SILT_MUD = "silt_mud"
    CONSTRUCTION_DEBRIS = "construction_debris"
    BLOCKED_DRAIN = "blocked_drain"
    STANDING_WATER = "standing_water"
    FLOWING_WATER = "flowing_water"
    OVERFLOWING_DRAIN = "overflowing_drain"
    ROAD_WATERLOGGING = "road_waterlogging"


# Grouping helpers for blockage and water detection
BLOCKAGE_CLASSES = {
    DrainWatchClass.GARBAGE.value,
    DrainWatchClass.PLASTIC_WASTE.value,
    DrainWatchClass.LEAVES.value,
    DrainWatchClass.SILT_MUD.value,
    DrainWatchClass.CONSTRUCTION_DEBRIS.value,
    DrainWatchClass.BLOCKED_DRAIN.value,
    "bottle", "cup", "handbag", "backpack", "suitcase",  # COCO proxies
}

DRAIN_STRUCTURE_CLASSES = {
    DrainWatchClass.DRAIN_INLET.value,
    DrainWatchClass.OPEN_DRAIN.value,
    DrainWatchClass.COVERED_DRAIN.value,
}

WATER_CLASSES = {
    DrainWatchClass.STANDING_WATER.value,
    DrainWatchClass.FLOWING_WATER.value,
    DrainWatchClass.OVERFLOWING_DRAIN.value,
    DrainWatchClass.ROAD_WATERLOGGING.value,
}


@dataclass
class DetectionBox:
    """Bounding box detection with class and confidence."""
    class_name: str
    confidence: float
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixel coordinates
    area: float                             # pixel area: (x2 - x1) * (y2 - y1)
    is_in_roi: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameObservation:
    """Structured vision observation extracted from a single camera frame."""
    camera_id: str
    timestamp: datetime
    frame_index: int
    garbage_detected: bool
    garbage_area: float                  # raw pixel area within ROI / active zone
    drain_area: float                    # detected drain inlet / open drain area (pixels)
    blockage_percentage: float           # estimated 0.0 to 100.0%
    water_coverage_percentage: float     # estimated 0.0 to 100.0%
    overflow_detected: bool
    confidence: float                    # aggregate detection confidence (0.0 to 1.0)
    condition_class: str                 # e.g. "drain_blocked_garbage", "drain_overflowing", "drain_clear"
    condition_score: float               # 0.0 (clear/safe) to 1.0 (critical blockage/overflow)
    detections: List[DetectionBox] = field(default_factory=list)
    # Absolute water depth in cm requires calibrated stereoscopic camera or reference markers.
    # This field is ALWAYS None unless camera calibration data is explicitly provided.
    water_depth_cm: Optional[float] = None
    calibration_required: bool = True
    # Normalized areas: fraction of inspection zone (ROI or frame) covered [0.0, 1.0]
    normalized_blockage_area: float = 0.0
    normalized_water_area: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert observation to dictionary serialization."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class AggregatedObservation:
    """
    Temporally aggregated observations across a sliding window of frames.

    Key design principles:
    - Persistence voting: garbage/overflow only flagged if present in >= 20% of frames
    - Confidence-weighted metrics: higher-confidence frames have proportionally more weight
    - Water-rise rate is Δ(water_coverage_%) per minute — a relative proxy, NOT absolute depth
    - is_valid=False when no observations available (fallback: use rainfall + terrain features only)
    """
    camera_id: str
    start_time: datetime
    end_time: datetime
    total_frames_analyzed: int
    garbage_detected: bool
    garbage_area_mean: float
    drain_area_mean: float
    blockage_percentage_mean: float
    water_coverage_percentage_mean: float
    overflow_detected: bool
    confidence_mean: float
    dominant_condition_class: str
    condition_score_mean: float
    staleness_hours: float
    is_valid: bool = True
    # Temporal velocity: Δwater_coverage_% per minute (positive = rising, negative = draining)
    # This is a relative proxy only — absolute depth requires calibrated cameras.
    water_rise_rate_pct_per_min: float = 0.0
    # Confidence-weighted condition score (weights = per-frame detection confidence)
    condition_score_confidence_weighted: float = 0.0
    # Observation window duration in minutes
    observation_window_minutes: float = 0.0
    # Number of frames where overflow was detected
    overflow_frame_count: int = 0

    def to_ml_features(self) -> Dict[str, Any]:
        """
        Format metrics for Stage 2 DrainWatch ML risk score adjustment.
        Missing CCTV fallback: when is_valid=False, all drainwatch_* values are neutral
        and drainwatch_available=False signals the ML layer to rely on rainfall+terrain only.
        """
        if not self.is_valid:
            return {
                "drainwatch_score": 0.0,
                "drainwatch_observation_class": "unavailable",
                "drainwatch_confidence": 0.0,
                "drainwatch_data_age_hours": self.staleness_hours,
                "drainwatch_available": False,
                "drainwatch_blockage_pct": 0.0,
                "drainwatch_water_coverage_pct": 0.0,
                "drainwatch_overflow_detected": False,
                "drainwatch_water_rise_rate": 0.0,
            }

        return {
            "drainwatch_score": round(self.condition_score_confidence_weighted or self.condition_score_mean, 3),
            "drainwatch_observation_class": self.dominant_condition_class,
            "drainwatch_confidence": round(self.confidence_mean, 3),
            "drainwatch_data_age_hours": round(self.staleness_hours, 2),
            "drainwatch_available": True,
            "drainwatch_blockage_pct": round(self.blockage_percentage_mean, 1),
            "drainwatch_water_coverage_pct": round(self.water_coverage_percentage_mean, 1),
            "drainwatch_overflow_detected": self.overflow_detected,
            # Relative water-rise rate proxy (% per minute); NOT absolute depth
            "drainwatch_water_rise_rate": round(self.water_rise_rate_pct_per_min, 3),
        }
