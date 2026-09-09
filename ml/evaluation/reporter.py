"""
Evaluation Report Generator with Spatial, Evidence-Tier, and CCTV Disaggregation.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from ml.evaluation.metrics import MetricSummary, calculate_metrics


class EvaluationReporter:
    """
    Generates structured and disaggregated evaluation reports.
    """

    @classmethod
    def evaluate_dataset(
        cls,
        df: pd.DataFrame,
        target_col: str = "road_inaccessible",
        prob_col: str = "calibrated_prob",
        zone_col: str = "zone_no",
        evidence_col: str = "primary_evidence_type",
        dw_avail_col: str = "drainwatch_available",
    ) -> Dict[str, Any]:
        """
        Evaluate predictions across overall dataset and disaggregated subsets.
        """
        report: Dict[str, Any] = {}

        y_all = df[target_col].values
        p_all = df[prob_col].values

        # Overall Metrics
        report["overall"] = calculate_metrics(y_all, p_all)

        # Spatial Disaggregation
        if zone_col in df.columns:
            spatial_metrics: Dict[str, MetricSummary] = {}
            for z, group in df.groupby(zone_col):
                if len(group) >= 5:
                    spatial_metrics[str(z)] = calculate_metrics(
                        group[target_col].values, group[prob_col].values
                    )
            report["by_zone"] = spatial_metrics

        # Evidence Tier Disaggregation
        if evidence_col in df.columns:
            evidence_metrics: Dict[str, MetricSummary] = {}
            for ev, group in df.groupby(evidence_col):
                if len(group) >= 5:
                    evidence_metrics[str(ev)] = calculate_metrics(
                        group[target_col].values, group[prob_col].values
                    )
            report["by_evidence_tier"] = evidence_metrics

        # CCTV Coverage Disaggregation
        if dw_avail_col in df.columns:
            cctv_metrics: Dict[str, MetricSummary] = {}
            with_cam = df[df[dw_avail_col].fillna(False).astype(bool)]
            without_cam = df[~df[dw_avail_col].fillna(False).astype(bool)]

            if len(with_cam) >= 3:
                cctv_metrics["with_cctv"] = calculate_metrics(
                    with_cam[target_col].values, with_cam[prob_col].values
                )
            if len(without_cam) >= 3:
                cctv_metrics["without_cctv"] = calculate_metrics(
                    without_cam[target_col].values, without_cam[prob_col].values
                )
            report["by_cctv_coverage"] = cctv_metrics

        return report

    @staticmethod
    def format_text_report(report: Dict[str, Any]) -> str:
        """
        Format technical evaluation summary for display and reporting.
        """
        ov: Optional[MetricSummary] = report.get("overall")
        if not ov:
            return "No overall metrics available."

        lines = [
            "==================================================",
            "  CHENNAI FLOOD ACCESS RISK - EVALUATION REPORT   ",
            "==================================================",
            f"Evaluated Segments:    {ov.n_samples} (Inaccessible: {ov.n_positives}, {ov.prevalence*100:.1f}%)",
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
        """Backward-compatible alias for evaluate_dataset."""
        return cls.evaluate_dataset(
            df=df,
            target_col=target_col,
            prob_col=prob_col,
            zone_col=zone_col,
            evidence_col=evidence_col,
            dw_avail_col=dw_avail_col,
        )

    @staticmethod
    def evaluate_predictions(*args, **kwargs) -> MetricSummary:
        """Backward-compatible alias for calculate_metrics."""
        return calculate_metrics(*args, **kwargs)

    @staticmethod
    def format_defensibility_summary(*args, **kwargs) -> str:
        """Backward-compatible alias for format_text_report."""
        return EvaluationReporter.format_text_report(*args, **kwargs)


# Alias
ModelEvaluator = EvaluationReporter
