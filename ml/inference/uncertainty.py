"""
Conformal Prediction and Epistemic Uncertainty Estimation for Risk Inference.
Outputs standard risk classes: LOW, MODERATE, HIGH, CRITICAL.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


RISK_BAND_NAMES = ["LOW", "MODERATE", "HIGH", "CRITICAL"]


def score_to_band(score: float) -> str:
    """
    Convert a continuous 0-100 risk score into a risk category band:
    - [0, 25): LOW
    - [25, 50): MODERATE
    - [50, 75): HIGH
    - [75, 100]: CRITICAL
    """
    if score < 25.0:
        return "LOW"
    elif score < 50.0:
        return "MODERATE"
    elif score < 75.0:
        return "HIGH"
    else:
        return "CRITICAL"


class ConformalRiskPredictor:
    """
    Inductive Conformal Prediction (ICP) producing valid prediction sets over
    LOW, MODERATE, HIGH, CRITICAL.
    """

    def __init__(self, coverage_level: float = 0.90) -> None:
        self.coverage_level = coverage_level
        self.alpha = 1.0 - coverage_level
        self.conformal_quantile: float = 0.25
        self.is_calibrated: bool = False

    def fit_calibration_set(
        self,
        predicted_probs: np.ndarray,
        true_labels: np.ndarray,
    ) -> "ConformalRiskPredictor":
        """
        Calibrate non-conformity threshold on holdout validation data.
        """
        p = np.asarray(predicted_probs, dtype=float)
        y = np.asarray(true_labels, dtype=float)

        valid = ~np.isnan(p) & ~np.isnan(y)
        if np.sum(valid) < 5:
            self.conformal_quantile = 0.35
            self.is_calibrated = True
            return self

        residuals = np.abs(y[valid] - p[valid])
        n = len(residuals)
        level = min(1.0, (n + 1) * (1.0 - self.alpha) / n)
        self.conformal_quantile = float(np.quantile(residuals, level, method="higher"))
        self.is_calibrated = True
        return self

    def predict_prediction_set(
        self,
        predicted_prob: float,
        extrapolation_multiplier: float = 1.0,
    ) -> List[str]:
        """
        Produce conformal prediction set of risk bands.
        """
        p = np.clip(predicted_prob, 0.0, 1.0)
        q = min(0.5, self.conformal_quantile * extrapolation_multiplier)

        lower_score = max(0.0, p - q) * 100.0
        upper_score = min(1.0, p + q) * 100.0

        bands_in_set: List[str] = []
        for band, (b_low, b_high) in [
            ("LOW", (0.0, 25.0)),
            ("MODERATE", (25.0, 50.0)),
            ("HIGH", (50.0, 75.0)),
            ("CRITICAL", (75.0, 100.0)),
        ]:
            if not (upper_score < b_low or lower_score > b_high):
                bands_in_set.append(band)

        return bands_in_set if bands_in_set else [score_to_band(p * 100.0)]

    def enrich_with_uncertainty(
        self,
        df: pd.DataFrame,
        prob_col: str = "calibrated_prob",
        hist_conf_col: str = "historical_confidence",
        dw_avail_col: str = "drainwatch_available",
        dw_age_col: str = "drainwatch_data_age_hours",
        extrapolation: bool = False,
    ) -> pd.DataFrame:
        """
        Add uncertainty indicators and conformal prediction sets to output table.
        """
        out = df.copy()
        n = len(out)

        mult = 1.2 if extrapolation else 1.0
        prediction_sets: List[List[str]] = []
        set_widths: List[int] = []
        is_uncertain_flags: List[bool] = []

        probs = out[prob_col].values if prob_col in out.columns else np.full(n, 0.5)

        for i in range(n):
            p = probs[i]
            if pd.isna(p):
                p = 0.5
            p_set = self.predict_prediction_set(p, extrapolation_multiplier=mult)
            prediction_sets.append(p_set)
            set_widths.append(len(p_set))
            is_uncertain_flags.append(bool(len(p_set) >= 2))

        out["prediction_set"] = prediction_sets
        out["prediction_set_width"] = set_widths
        out["is_uncertain"] = is_uncertain_flags
        out["scenario_extrapolation"] = extrapolation

        return out
