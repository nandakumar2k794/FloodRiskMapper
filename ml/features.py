"""
ML Module: Road segment feature engineering interface.
"""
from typing import Dict, Any, List


class FeatureExtractor:
    """Interface for combining rainfall, terrain, and drainage features per road segment."""

    def __init__(self) -> None:
        pass

    def extract_features(
        self,
        segment_id: str,
        rainfall_24h_mm: float,
        hand_m: float,
        slope_pct: float,
        drainwatch_score: float,
        historical_freq: float,
    ) -> Dict[str, float]:
        """Combine raw physical and visual metrics into model feature vector."""
        raise NotImplementedError("Phase 4 will implement feature engineering pipeline.")

    def batch_extract(self, road_segments: List[Dict[str, Any]]) -> Any:
        """Vectorized feature extraction across all road segments."""
        raise NotImplementedError("Phase 4 will implement batch feature extraction.")
