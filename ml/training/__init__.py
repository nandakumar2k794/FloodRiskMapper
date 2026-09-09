"""
ML Training Subpackage.
Contains model trainers, spatial & temporal splitting, calibration, and serialization.
"""
from ml.training.split import (
    SpatialZoneSplitter,
    TemporalSplitter,
    SPATIAL_SPLIT_ZONES,
)
from ml.training.calibration import (
    calculate_ece,
    compute_reliability_curve,
    IsotonicProbabilityCalibrator,
)
from ml.training.trainer import (
    FloodRiskTrainer,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_FILENAME,
    MONOTONIC_INCREASING_FEATURES,
)

__all__ = [
    "SpatialZoneSplitter",
    "TemporalSplitter",
    "SPATIAL_SPLIT_ZONES",
    "calculate_ece",
    "compute_reliability_curve",
    "IsotonicProbabilityCalibrator",
    "FloodRiskTrainer",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_MODEL_FILENAME",
    "MONOTONIC_INCREASING_FEATURES",
]
