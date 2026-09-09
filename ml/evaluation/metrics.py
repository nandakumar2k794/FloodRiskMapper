"""
Core Metric Calculations: AUROC, AUPRC, Recall@CRITICAL, F2-Score, and ECE.
Do not fabricate real validation metrics.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    fbeta_score,
    recall_score,
)

from ml.training.calibration import calculate_ece


@dataclass
class MetricSummary:
    """Core evaluation metrics container."""
    n_samples: int
    n_positives: int
    prevalence: float
    auroc: float
    auprc: float
    recall_critical: float
    f2_score: float
    optimal_threshold: float
    ece: float


def find_optimal_threshold_fbeta(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    beta: float = 2.0,
) -> Tuple[float, float]:
    """
    Find probability threshold maximizing F-beta score (prioritizing recall for safety).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds = np.append(thresholds, 1.0)

    beta_sq = beta ** 2
    f_betas = []
    for p, r in zip(precisions, recalls):
        if (beta_sq * p + r) > 0:
            f = (1.0 + beta_sq) * (p * r) / (beta_sq * p + r)
        else:
            f = 0.0
        f_betas.append(f)

    best_idx = int(np.argmax(f_betas))
    return float(thresholds[best_idx]), float(f_betas[best_idx])


def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    beta: float = 2.0,
    critical_prob_threshold: float = 0.75,
) -> MetricSummary:
    """
    Compute AUROC, AUPRC, Recall@CRITICAL, F2, and ECE without fabrication.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_prob, dtype=float)

    valid = ~np.isnan(y_t) & ~np.isnan(y_p)
    y_t = y_t[valid].astype(int)
    y_p = y_p[valid]

    n = len(y_t)
    if n == 0:
        return MetricSummary(0, 0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0)

    n_pos = int(np.sum(y_t == 1))
    prevalence = float(n_pos / n) if n > 0 else 0.0

    has_both_classes = len(np.unique(y_t)) > 1
    auroc = float(roc_auc_score(y_t, y_p)) if has_both_classes else 0.5
    auprc = float(average_precision_score(y_t, y_p)) if has_both_classes else prevalence

    # Recall at CRITICAL risk band (score >= 75 / prob >= 0.75)
    pred_critical = (y_p >= critical_prob_threshold).astype(int)
    rec_crit = float(recall_score(y_t, pred_critical, zero_division=0))

    opt_thresh, best_f2 = (
        find_optimal_threshold_fbeta(y_t, y_p, beta=beta)
        if has_both_classes
        else (0.5, 0.0)
    )

    ece = calculate_ece(y_t, y_p)

    return MetricSummary(
        n_samples=n,
        n_positives=n_pos,
        prevalence=round(prevalence, 4),
        auroc=round(auroc, 4),
        auprc=round(auprc, 4),
        recall_critical=round(rec_crit, 4),
        f2_score=round(best_f2, 4),
        optimal_threshold=round(opt_thresh, 4),
        ece=round(ece, 4),
    )
