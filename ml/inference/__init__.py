"""
ML Inference Subpackage.
Contains risk scoring, DrainWatch CCTV adjustments, conformal uncertainty, and vehicle modulation.
"""
from ml.inference.scorer import (
    RoadFloodRiskScorer,
    compute_vehicle_risk_scores,
    VEHICLE_RISK_OFFSETS,
    MAX_TRAINING_RAINFALL_24H_MM,
)
from ml.inference.drainwatch import (
    DrainWatchAdjuster,
    CLASS_DELTA_POINTS,
    get_staleness_multiplier,
)
from ml.inference.uncertainty import (
    ConformalRiskPredictor,
    score_to_band,
    RISK_BAND_NAMES,
)
from ml.explanation import (
    RiskExplanationGenerator,
    RoadRiskExplanation,
    RiskFactor,
)
from ml.fusion import (
    DrainWatchFusionEngine,
    get_fusion_engine,
    fuse_drainwatch_risk,
)

__all__ = [
    "RoadFloodRiskScorer",
    "compute_vehicle_risk_scores",
    "VEHICLE_RISK_OFFSETS",
    "MAX_TRAINING_RAINFALL_24H_MM",
    "DrainWatchAdjuster",
    "CLASS_DELTA_POINTS",
    "get_staleness_multiplier",
    "ConformalRiskPredictor",
    "score_to_band",
    "RISK_BAND_NAMES",
    "DrainWatchFusionEngine",
    "get_fusion_engine",
    "fuse_drainwatch_risk",
    "RiskExplanationGenerator",
    "RoadRiskExplanation",
    "RiskFactor",
]
