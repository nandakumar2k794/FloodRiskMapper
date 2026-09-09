"""
ML Evaluation Subpackage.
Contains core metrics, threshold optimization, and disaggregated reporting.
"""
from ml.evaluation.metrics import (
    MetricSummary,
    find_optimal_threshold_fbeta,
    calculate_metrics,
)
from ml.evaluation.reporter import EvaluationReporter, ModelEvaluator

__all__ = [
    "MetricSummary",
    "find_optimal_threshold_fbeta",
    "calculate_metrics",
    "EvaluationReporter",
    "ModelEvaluator",
]
