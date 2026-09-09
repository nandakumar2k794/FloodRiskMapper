"""
Unit tests for Model Evaluation and Defensibility Reporting (Phase 4d).
Tests:
- F-beta (F2) optimal threshold selection prioritizing recall
- Primary metric computation (AUROC, AUPRC, Recall@CRITICAL, ECE)
- Disaggregated evaluations across zones, evidence tiers, and CCTV presence
- Defensibility report text generation
"""
import pytest
import numpy as np
import pandas as pd

from ml.evaluation import (
    ModelEvaluator,
    find_optimal_threshold_fbeta,
    MetricSummary,
)


@pytest.fixture
def sample_eval_df():
    """Create a mock prediction evaluation DataFrame."""
    np.random.seed(42)
    n = 100

    y_true = np.random.choice([0, 1], size=n, p=[0.75, 0.25])
    # Add noise to true label to create probabilities
    noise = np.random.uniform(-0.2, 0.2, n)
    y_prob = np.clip(y_true * 0.7 + 0.15 + noise, 0.0, 1.0)

    zones = np.random.choice(["ZONE_01", "ZONE_05", "ZONE_13"], size=n)
    evidence = np.random.choice(["OBSERVED", "INFERRED", "SYNTHETIC"], size=n)
    cctv = np.random.choice([True, False], size=n, p=[0.2, 0.8])

    return pd.DataFrame({
        "road_inaccessible": y_true,
        "calibrated_prob": y_prob,
        "zone_no": zones,
        "primary_evidence_type": evidence,
        "drainwatch_available": cctv,
    })


class TestModelEvaluation:
    """Tests for ModelEvaluator."""

    def test_optimal_threshold_fbeta(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.6, 0.8, 0.9])

        thresh, f2 = find_optimal_threshold_fbeta(y_true, y_prob, beta=2.0)
        assert 0.0 <= thresh <= 1.0
        assert f2 > 0.7  # should achieve strong F2 score

    def test_evaluate_predictions_metrics(self, sample_eval_df):
        y_true = sample_eval_df["road_inaccessible"].values
        y_prob = sample_eval_df["calibrated_prob"].values

        summary = ModelEvaluator.evaluate_predictions(y_true, y_prob)

        assert isinstance(summary, MetricSummary)
        assert summary.n_samples == 100
        assert 0.5 <= summary.auroc <= 1.0
        assert 0.0 <= summary.auprc <= 1.0
        assert 0.0 <= summary.recall_critical <= 1.0
        assert 0.0 <= summary.ece <= 1.0

    def test_evaluate_disaggregated(self, sample_eval_df):
        report = ModelEvaluator.evaluate_disaggregated(sample_eval_df)

        assert "overall" in report
        assert "by_zone" in report
        assert "by_evidence_tier" in report
        assert "by_cctv_coverage" in report

        # Check zone breakdown
        for z, metrics in report["by_zone"].items():
            assert metrics.n_samples > 0
            assert 0.0 <= metrics.auroc <= 1.0

        # Check evidence breakdown
        for ev, metrics in report["by_evidence_tier"].items():
            assert ev in ["OBSERVED", "INFERRED", "SYNTHETIC"]
            assert metrics.n_samples > 0

    def test_defensibility_summary_formatting(self, sample_eval_df):
        report = ModelEvaluator.evaluate_disaggregated(sample_eval_df)
        summary_text = ModelEvaluator.format_defensibility_summary(report)

        assert "CHENNAI FLOOD ACCESS RISK - EVALUATION REPORT" in summary_text
        assert "AUROC (Discrimination)" in summary_text
        assert "Calibration Error (ECE)" in summary_text
        assert "By Evidence Tier:" in summary_text
