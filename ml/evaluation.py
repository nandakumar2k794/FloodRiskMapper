"""
Model Evaluation and Defensibility Reporting Module.
Implements §13 of the ML methodology:
- Computes AUROC, AUPRC, Recall@CRITICAL, ECE, and F1-beta (beta=2).
- Optimizes decision threshold prioritizing recall for emergency response.
- Generates disaggregated metrics across:
    1. Spatial zones (e.g. North vs South Chennai).
    2. Evidence tiers (OBSERVED vs INFERRED vs SYNTHETIC).
    3. CCTV coverage status (DrainWatch available vs unavailable).
- Formats technically defensible evaluation reports for hackathon presentation.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    fbeta_score,
    recall_score,
    precision_score,
    confusion_matrix,
)

from ml.calibration import calculate_ece


@dataclass
class MetricSummary:
    """Core performance metrics for a specific subset."""
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
    Find probability threshold maximizing F-beta score (prioritizing recall).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Exclude endpoint where threshold is 1.0
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


class ModelEvaluator:
    """
    Calculates comprehensive, disaggregated evaluation metrics.
    """

    @staticmethod
    def evaluate_predictions(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        beta: float = 2.0,
        critical_prob_threshold: float = 0.75,
    ) -> MetricSummary:
        """
        Compute primary metrics on binary ground truth and predicted probabilities.
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

        # Optimal F-beta threshold
        opt_thresh, best_f2 = find_optimal_threshold_fbeta(y_t, y_p, beta=beta) if has_both_classes else (0.5, 0.0)

        # Expected Calibration Error
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

    @classmethod
    def evaluate_disaggregated(
        cls,
        df: pd.DataFrame,
        target_col: str = "road_inaccessible",
        prob_col: str = "calibrated_prob",
        zone_col: str = "zone_no",
        evidence_col: str = "primary_evidence_type",
        dw_avail_col: str = "drainwatch_available",
    ) -> Dict[str, Any]:
        """
        Compute overall and disaggregated evaluation metrics across space,
        evidence tiers, and CCTV coverage.
        """
        report: Dict[str, Any] = {}

        y_all = df[target_col].values
        p_all = df[prob_col].values

        # 1. Overall Metrics
        report["overall"] = cls.evaluate_predictions(y_all, p_all)

        # 2. Spatial Disaggregation
        if zone_col in df.columns:
            spatial_metrics: Dict[str, MetricSummary] = {}
            for z, group in df.groupby(zone_col):
                if len(group) >= 5:
                    spatial_metrics[str(z)] = cls.evaluate_predictions(
                        group[target_col].values, group[prob_col].values
                    )
            report["by_zone"] = spatial_metrics

        # 3. Evidence Tier Disaggregation
        if evidence_col in df.columns:
            evidence_metrics: Dict[str, MetricSummary] = {}
            for ev, group in df.groupby(evidence_col):
                if len(group) >= 5:
                    evidence_metrics[str(ev)] = cls.evaluate_predictions(
                        group[target_col].values, group[prob_col].values
                    )
            report["by_evidence_tier"] = evidence_metrics

        # 4. CCTV Coverage Disaggregation
        if dw_avail_col in df.columns:
            cctv_metrics: Dict[str, MetricSummary] = {}
            with_cam = df[df[dw_avail_col].fillna(False).astype(bool)]
            without_cam = df[~df[dw_avail_col].fillna(False).astype(bool)]

            if len(with_cam) >= 3:
                cctv_metrics["with_cctv"] = cls.evaluate_predictions(
                    with_cam[target_col].values, with_cam[prob_col].values
                )
            if len(without_cam) >= 3:
                cctv_metrics["without_cctv"] = cls.evaluate_predictions(
                    without_cam[target_col].values, without_cam[prob_col].values
                )
            report["by_cctv_coverage"] = cctv_metrics

        return report

    @staticmethod
    def format_defensibility_summary(report: Dict[str, Any]) -> str:
        """
        Format technical evaluation summary string for hackathon presentation.
        """
        ov = report.get("overall")
        if not ov:
            return "No overall metrics available."

        lines = [
            "==================================================",
            "  CHENNAI FLOOD ACCESS RISK - EVALUATION REPORT   ",
            "==================================================",
            f"Evaluated Samples:     {ov.n_samples} (Inaccessible: {ov.n_positives}, {ov.prevalence*100:.1f}%)",
            f"AUROC (Discrimination): {ov.auroc:.3f}  [Expected: 0.72 - 0.85]",
            f"AUPRC (Precision/Rec):  {ov.auprc:.3f}",
            f"Recall @ CRITICAL band: {ov.recall_critical*100:.1f}%",
            f"Optimal F2-Score:       {ov.f2_score:.3f}  (Threshold: {ov.optimal_threshold:.2f})",
            f"Calibration Error (ECE):{ov.ece:.4f}  [Target < 0.08: {'PASSED' if ov.ece < 0.08 else 'FAILED'}]",
            "--------------------------------------------------",
            "Disaggregation Highlights:",
        ]

        if "by_evidence_tier" in report:
            lines.append("  By Evidence Tier:")
            for tier, m in report["by_evidence_tier"].items():
                lines.append(f"    - {tier} (N={m.n_samples}): AUROC={m.auroc:.3f}, ECE={m.ece:.3f}")

        if "by_cctv_coverage" in report:
            lines.append("  By CCTV DrainWatch Coverage:")
            for status, m in report["by_cctv_coverage"].items():
                lines.append(f"    - {status} (N={m.n_samples}): AUROC={m.auroc:.3f}, F2={m.f2_score:.3f}")

        lines.append("==================================================")
        return "\n".join(lines)
