"""
ML Preprocessing Subpackage.
Contains feature engineering, labeling, evidence weighting, and leakage audit.
"""
from ml.preprocessing.features import (
    FeatureMatrixBuilder,
    MODEL_FEATURE_NAMES,
    CATEGORICAL_FEATURES,
    DRAINWATCH_FEATURE_NAMES,
)
from ml.preprocessing.labeling import (
    generate_real_labels,
    generate_synthetic_scenario_labels,
)
from ml.preprocessing.weights import (
    calculate_scale_pos_weight,
    compute_sample_weights,
)
from ml.preprocessing.leakage import (
    DataLeakageAuditor,
    DataLeakageError,
    LeakageAuditReport,
)

__all__ = [
    "FeatureMatrixBuilder",
    "MODEL_FEATURE_NAMES",
    "CATEGORICAL_FEATURES",
    "DRAINWATCH_FEATURE_NAMES",
    "generate_real_labels",
    "generate_synthetic_scenario_labels",
    "calculate_scale_pos_weight",
    "compute_sample_weights",
    "DataLeakageAuditor",
    "DataLeakageError",
    "LeakageAuditReport",
]
