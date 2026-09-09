"""
Stage 2 CCTV DrainWatch Additive Adjustment Module for Inference.
"""
from ml.drainwatch_adjustment import (
    CLASS_DELTA_POINTS,
    get_staleness_multiplier,
    DrainWatchAdjuster,
)

__all__ = [
    "CLASS_DELTA_POINTS",
    "get_staleness_multiplier",
    "DrainWatchAdjuster",
]
