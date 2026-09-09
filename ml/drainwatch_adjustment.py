"""
DrainWatch CCTV Stage 2 Additive Adjustment Module.
Implements §15 of the ML methodology:
- Two-stage architecture: Base model scores all roads; Stage 2 applies an additive delta
  only for segments with verified CCTV camera observations.
- Temporal staleness decay protocol:
    0–60 min: 1.0 (full weight)
    1–4 hours: 0.7
    4–12 hours: 0.3
    > 12 hours: 0.0 (ignored, zero delta)
- Road segments without CCTV coverage maintain a delta of 0.0 (base score preserved).
"""
from typing import Optional, Union, Dict, Any
import numpy as np
import pandas as pd

from config.constants import DRAINWATCH_CLASS_SCORES


# Class delta lookup table (points to add/subtract from base score 0-100)
CLASS_DELTA_POINTS: Dict[str, float] = {
    "road_waterlogged": 30.0,
    "drain_overflowing": 20.0,
    "standing_water_major": 18.0,
    "drain_blocked_garbage": 15.0,
    "drain_blocked_silt": 10.0,
    "standing_water_minor": 8.0,
    "drain_clear": -5.0,
}


def get_staleness_multiplier(age_hours: Optional[float], observation_class: Optional[str] = None) -> float:
    """
    Calculate temporal staleness decay multiplier.
    
    Fast decay (water, overflow):
        0 - 1h:   1.0
        1 - 4h:   0.7
        4 - 12h:  0.3
        > 12h:    0.0 (or NaN age -> 0.0)
        
    Slow decay (garbage/silt blockage):
        0 - 12h:  1.0
        12 - 48h: 0.5
        48 - 168h: 0.2
        > 168h:   0.0
    """
    if age_hours is None or pd.isna(age_hours) or age_hours < 0:
        return 0.0

    obs = str(observation_class).strip().lower() if observation_class else ""
    is_slow_decay = obs in ["drain_blocked_garbage", "drain_blocked_silt"]

    if is_slow_decay:
        if age_hours <= 12.0:
            return 1.0
        elif age_hours <= 48.0:
            return 0.5
        elif age_hours <= 168.0:
            return 0.2
        else:
            return 0.0
    else:
        if age_hours <= 1.0:
            return 1.0
        elif age_hours <= 4.0:
            return 0.7
        elif age_hours <= 12.0:
            return 0.3
        else:
            return 0.0


class DrainWatchAdjuster:
    """
    Stage 2 updater applying vision-derived CCTV evidence deltas to base road risk scores.
    """

    def __init__(self, max_positive_delta: float = 30.0, max_negative_delta: float = -10.0) -> None:
        self.max_positive_delta = max_positive_delta
        self.max_negative_delta = max_negative_delta

    def compute_single_delta(
        self,
        observation_class: Optional[str] = None,
        confidence: Optional[float] = 1.0,
        age_hours: Optional[float] = 0.0,
        drainwatch_score: Optional[float] = None,
        blockage_pct: Optional[float] = None,
        water_coverage_pct: Optional[float] = None,
        overflow_detected: Optional[bool] = None,
        water_rise_rate: Optional[float] = None,
    ) -> float:
        """
        Compute additive delta for a single segment's observation.
        Considers CCTV observation class, detection confidence, temporal staleness,
        and continuous vision metrics (blockage %, water rise rate).
        """
        # Temporal staleness check
        staleness_mult = get_staleness_multiplier(age_hours, observation_class)
        if staleness_mult <= 0.0:
            return 0.0

        # Confidence weighting (noise filter: confidence < 0.2 is suppressed)
        conf_val = float(confidence) if confidence is not None and not pd.isna(confidence) else 1.0
        if conf_val < 0.2:
            return 0.0
        conf_mult = float(np.clip(conf_val, 0.0, 1.0))

        # Base raw delta determination
        if observation_class is None or pd.isna(observation_class):
            if drainwatch_score is not None and not pd.isna(drainwatch_score):
                # Infer delta from continuous score in [0.0, 1.0]
                raw_delta = (float(drainwatch_score) - 0.2) * 25.0
            else:
                raw_delta = 0.0
        else:
            obs_clean = str(observation_class).strip().lower()
            raw_delta = CLASS_DELTA_POINTS.get(obs_clean, 0.0)

        # Refine with continuous indicators when available
        if overflow_detected is True or (isinstance(overflow_detected, (int, float)) and overflow_detected > 0):
            raw_delta = max(raw_delta, 20.0)

        if blockage_pct is not None and not pd.isna(blockage_pct):
            b_pct = float(blockage_pct)
            if b_pct >= 50.0 and raw_delta < 15.0:
                raw_delta = 15.0
            elif b_pct >= 30.0 and raw_delta < 10.0:
                raw_delta = 10.0

        if water_coverage_pct is not None and not pd.isna(water_coverage_pct):
            w_cov = float(water_coverage_pct)
            if w_cov >= 60.0 and raw_delta < 25.0:
                raw_delta = 25.0

        if water_rise_rate is not None and not pd.isna(water_rise_rate):
            w_rise = float(water_rise_rate)
            if w_rise > 0.5:
                # Dynamic velocity adjustment: up to +6 points for rapidly rising water
                raw_delta += min(6.0, w_rise * 1.5)

        final_delta = raw_delta * staleness_mult * conf_mult
        return float(np.clip(final_delta, self.max_negative_delta, self.max_positive_delta))

    def apply_adjustments(
        self,
        df: pd.DataFrame,
        base_score_col: str = "base_risk_score",
        obs_class_col: str = "drainwatch_observation_class",
        confidence_col: str = "drainwatch_confidence",
        age_hours_col: str = "drainwatch_data_age_hours",
        available_col: str = "drainwatch_available",
        output_col: str = "adjusted_risk_score",
        delta_col: str = "drainwatch_delta",
    ) -> pd.DataFrame:
        """
        Vectorized application of DrainWatch Stage 2 adjustments across road segments.

        Parameters:
            df: DataFrame containing base risk scores and DrainWatch feature columns.

        Returns:
            pd.DataFrame: Enriched DataFrame with `drainwatch_delta` and `adjusted_risk_score`.
        """
        out = df.copy()
        n = len(out)
        deltas = np.zeros(n, dtype=float)

        base_scores = out[base_score_col].values if base_score_col in out.columns else np.zeros(n)
        is_avail = (
            out[available_col].fillna(False).astype(bool).values
            if available_col in out.columns
            else np.zeros(n, dtype=bool)
        )
        classes = out[obs_class_col].values if obs_class_col in out.columns else [None] * n
        confs = out[confidence_col].values if confidence_col in out.columns else [1.0] * n
        ages = out[age_hours_col].values if age_hours_col in out.columns else [0.0] * n

        dw_scores = out["drainwatch_score"].values if "drainwatch_score" in out.columns else [None] * n
        blockages = out["drainwatch_blockage_pct"].values if "drainwatch_blockage_pct" in out.columns else [None] * n
        water_covs = out["drainwatch_water_coverage_pct"].values if "drainwatch_water_coverage_pct" in out.columns else [None] * n
        overflows = out["drainwatch_overflow_detected"].values if "drainwatch_overflow_detected" in out.columns else [None] * n
        rise_rates = out["drainwatch_water_rise_rate"].values if "drainwatch_water_rise_rate" in out.columns else [None] * n

        for i in range(n):
            if is_avail[i]:
                deltas[i] = self.compute_single_delta(
                    observation_class=classes[i],
                    confidence=confs[i],
                    age_hours=ages[i],
                    drainwatch_score=dw_scores[i],
                    blockage_pct=blockages[i],
                    water_coverage_pct=water_covs[i],
                    overflow_detected=overflows[i],
                    water_rise_rate=rise_rates[i],
                )
            else:
                deltas[i] = 0.0

        adjusted = np.clip(base_scores + deltas, 0.0, 100.0)

        out[delta_col] = np.round(deltas, 2)
        out[output_col] = np.round(adjusted, 1)

        return out
