"""
Sample weight assignment and class imbalance utilities.
Implements the multi-tiered sample weighting strategy defined in §9 of the ML methodology.
Separates OBSERVED, INFERRED, and SYNTHETIC evidence tiers.
"""
from typing import Optional, Union, Dict, Any
import numpy as np
import pandas as pd

from historical.models import HistoricalEvidenceType


# Evidence-tier weight constants for positive and negative observations
WEIGHT_OBSERVED_POSITIVE = 1.0
WEIGHT_INFERRED_POSITIVE = 0.7
WEIGHT_SYNTHETIC_POSITIVE = 0.5
WEIGHT_OBSERVED_NEGATIVE = 1.0
WEIGHT_NO_EVIDENCE_NEGATIVE = 0.3
WEIGHT_SYNTHETIC_NOISE_POSITIVE = 0.3


def calculate_scale_pos_weight(y: Union[np.ndarray, pd.Series, list]) -> float:
    """
    Calculate XGBoost scale_pos_weight (ratio of negative to positive samples).
    
    Formula:
        scale_pos_weight = count(negative) / count(positive)

    Parameters:
        y: Binary target array (0 = accessible, 1 = inaccessible). May contain NaNs.
    
    Returns:
        float: Scale pos weight (defaults to 1.0 if empty or zero positives).
    """
    s = pd.Series(y).dropna()
    pos = int((s == 1).sum())
    neg = int((s == 0).sum())

    if pos == 0:
        return 1.0
    return float(neg / pos)


def compute_sample_weights(
    df: pd.DataFrame,
    target_col: str = "road_inaccessible",
    evidence_col: str = "primary_evidence_type",
    source_col: str = "sample_source",
    is_noise_col: str = "is_synthetic_noise",
) -> np.ndarray:
    """
    Compute observation-level sample weights based on evidence tier and label.
    
    Weights:
        - Confirmed positive with OBSERVED evidence: 1.0
        - Positive with INFERRED evidence: 0.7
        - Positive with SYNTHETIC evidence: 0.5
        - Positive with injected synthetic noise: 0.3
        - Confirmed negative with positive evidence of passability: 1.0
        - Negative with no flood history / absence of evidence: 0.3
        - Unknown / NaN labels: 0.0 (masked out)

    Parameters:
        df: DataFrame containing features, labels, and evidence metadata.
        target_col: Name of target column (0, 1, or NaN).
        evidence_col: Column with evidence tier ('OBSERVED', 'INFERRED', 'SYNTHETIC', 'NONE').
        source_col: Column indicating 'REAL', 'SYNTHETIC', or 'SIMULATED'.
        is_noise_col: Optional boolean flag for synthetic noise samples.

    Returns:
        np.ndarray: Array of non-negative float weights matching df length.
    """
    n = len(df)
    weights = np.zeros(n, dtype=float)

    if n == 0:
        return weights

    targets = df[target_col].values if target_col in df.columns else np.full(n, np.nan)
    evidence = (
        df[evidence_col].astype(str).str.upper().values
        if evidence_col in df.columns
        else np.full(n, "NONE")
    )
    sources = (
        df[source_col].astype(str).str.upper().values
        if source_col in df.columns
        else np.full(n, "REAL")
    )
    has_noise = is_noise_col in df.columns
    noise_flags = df[is_noise_col].fillna(False).values if has_noise else np.zeros(n, dtype=bool)

    for i in range(n):
        y = targets[i]
        ev = evidence[i]
        src = sources[i]
        is_noise = noise_flags[i]

        if pd.isna(y):
            # Unknown label: weight = 0.0 (masked from supervised gradient)
            weights[i] = 0.0
        elif y == 1:
            # Positive sample: road was inaccessible
            if is_noise:
                weights[i] = WEIGHT_SYNTHETIC_NOISE_POSITIVE
            elif ev == "OBSERVED":
                weights[i] = WEIGHT_OBSERVED_POSITIVE
            elif ev == "INFERRED":
                weights[i] = WEIGHT_INFERRED_POSITIVE
            elif ev == "SYNTHETIC" or src == "SYNTHETIC":
                weights[i] = WEIGHT_SYNTHETIC_POSITIVE
            elif src == "REAL":
                weights[i] = WEIGHT_OBSERVED_POSITIVE
            else:
                weights[i] = WEIGHT_INFERRED_POSITIVE
        elif y == 0:
            # Negative sample: road was passable
            if ev == "OBSERVED":
                weights[i] = WEIGHT_OBSERVED_NEGATIVE
            else:
                # Absence of flood report does not guarantee complete passability
                weights[i] = WEIGHT_NO_EVIDENCE_NEGATIVE
        else:
            weights[i] = 0.0

    return weights
