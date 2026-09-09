"""
Probability Calibration and Reliability Assessment Module.
Implements §10 of the ML methodology:
- Isotonic regression calibration on spatial validation zone (Zones 10–12).
- Reliability curve calculation across 10 probability bins.
- Expected Calibration Error (ECE) measurement (target ECE < 0.08).
- Disaggregated calibration metrics by evidence tier (OBSERVED / INFERRED / SYNTHETIC).
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def calculate_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    Formula:
        ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Parameters:
        y_true: True binary targets (0 or 1).
        y_prob: Predicted probabilities in [0.0, 1.0].
        n_bins: Number of equal-width bins (default 10).

    Returns:
        float: ECE value in [0.0, 1.0].
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Filter out NaNs if any
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y_true = y_true[valid_mask]
    y_prob = y_prob[valid_mask]

    n = len(y_true)
    if n == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, List[float]]:
    """
    Compute reliability diagram points (predicted confidence vs empirical accuracy).

    Returns:
        Dict with 'bin_centers', 'mean_confidences', 'accuracies', 'counts'.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y_true = y_true[valid_mask]
    y_prob = y_prob[valid_mask]

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    mean_confidences = []
    accuracies = []
    counts = []

    for i in range(n_bins):
        b_low = bin_boundaries[i]
        b_high = bin_boundaries[i + 1]
        bin_centers.append(float((b_low + b_high) / 2.0))

        if i == n_bins - 1:
            in_bin = (y_prob >= b_low) & (y_prob <= b_high)
        else:
            in_bin = (y_prob >= b_low) & (y_prob < b_high)

        cnt = int(np.sum(in_bin))
        counts.append(cnt)
        if cnt > 0:
            mean_confidences.append(float(np.mean(y_prob[in_bin])))
            accuracies.append(float(np.mean(y_true[in_bin])))
        else:
            mean_confidences.append(float((b_low + b_high) / 2.0))
            accuracies.append(0.0)

    return {
        "bin_centers": bin_centers,
        "mean_confidences": mean_confidences,
        "accuracies": accuracies,
        "counts": counts,
    }


class IsotonicProbabilityCalibrator:
    """
    Calibrates uncalibrated model probabilities using isotonic regression
    with clipping to [0.0, 1.0].
    """

    def __init__(self, out_of_bounds: str = "clip") -> None:
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds=out_of_bounds)
        self.is_fitted = False

    def fit(self, uncalibrated_probs: np.ndarray, y_true: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> "IsotonicProbabilityCalibrator":
        """
        Fit isotonic regression calibrator.
        """
        probs = np.asarray(uncalibrated_probs, dtype=float)
        labels = np.asarray(y_true, dtype=float)

        valid = ~np.isnan(probs) & ~np.isnan(labels)
        if np.sum(valid) < 5:
            # Fallback: if not enough samples, mark unfitted
            self.is_fitted = False
            return self

        p_fit = probs[valid]
        y_fit = labels[valid]
        w_fit = np.asarray(sample_weights)[valid] if sample_weights is not None else None

        self.calibrator.fit(p_fit, y_fit, sample_weight=w_fit)
        self.is_fitted = True
        return self

    def calibrate(self, uncalibrated_probs: np.ndarray) -> np.ndarray:
        """
        Transform raw predicted probabilities into calibrated probabilities.
        """
        probs = np.asarray(uncalibrated_probs, dtype=float)
        if not self.is_fitted:
            return np.clip(probs, 0.0, 1.0)

        calibrated = self.calibrator.predict(probs)
        return np.clip(calibrated, 0.0, 1.0)

    def evaluate_calibration(
        self,
        uncalibrated_probs: np.ndarray,
        y_true: np.ndarray,
        evidence_types: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate calibration performance before and after calibration.
        """
        raw_ece = calculate_ece(y_true, uncalibrated_probs)
        calibrated_probs = self.calibrate(uncalibrated_probs)
        cal_ece = calculate_ece(y_true, calibrated_probs)

        report: Dict[str, Any] = {
            "raw_ece": raw_ece,
            "calibrated_ece": cal_ece,
            "meets_target": cal_ece < 0.08,
            "overall_curve": compute_reliability_curve(y_true, calibrated_probs),
        }

        # Disaggregate by evidence tier if provided
        if evidence_types is not None:
            disaggregated: Dict[str, Any] = {}
            evs = np.asarray(evidence_types, dtype=str)
            for tier in np.unique(evs):
                mask = evs == tier
                if np.sum(mask) >= 5:
                    tier_ece = calculate_ece(y_true[mask], calibrated_probs[mask])
                    disaggregated[tier] = {
                        "ece": tier_ece,
                        "sample_count": int(np.sum(mask)),
                    }
            report["evidence_tier_ece"] = disaggregated

        return report
