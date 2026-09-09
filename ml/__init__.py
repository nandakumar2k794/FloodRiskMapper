"""
Machine Learning and Flood Risk Modeling package for Chennai Flood Access Risk Mapper.
Subpackages:
- ml.preprocessing: Feature engineering, labeling, evidence weighting, leakage audit
- ml.training: XGBoost model training, spatial/temporal splitting, calibration, serialization
- ml.inference: Risk scoring, DrainWatch CCTV adjustments, conformal uncertainty, vehicle modulation
- ml.evaluation: Validation metrics, threshold optimization, disaggregated reports
"""
from ml.preprocessing import (
    FeatureMatrixBuilder,
    MODEL_FEATURE_NAMES,
    CATEGORICAL_FEATURES,
    DRAINWATCH_FEATURE_NAMES,
    generate_real_labels,
    generate_synthetic_scenario_labels,
    calculate_scale_pos_weight,
    compute_sample_weights,
    DataLeakageAuditor,
    DataLeakageError,
)
from ml.training import (
    FloodRiskTrainer,
    SpatialZoneSplitter,
    TemporalSplitter,
    IsotonicProbabilityCalibrator,
    calculate_ece,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_FILENAME,
)
from ml.inference import (
    RoadFloodRiskScorer,
    DrainWatchAdjuster,
    ConformalRiskPredictor,
    score_to_band,
    compute_vehicle_risk_scores,
    RISK_BAND_NAMES,
)
from ml.explanation import (
    RiskExplanationGenerator,
    RoadRiskExplanation,
    RiskFactor,
)
from ml.fusion import (
    DrainWatchFusionEngine,
    get_fusion_engine,
    fuse_drainwatch_risk,
)
from ml.evaluation import (
    MetricSummary,
    find_optimal_threshold_fbeta,
    calculate_metrics,
    EvaluationReporter,
)

# Compatibility placeholders from Phase 1
from ml.rainfall import RainfallService
from ml.features import FeatureExtractor
from ml.flood_model import RoadFloodRiskModel

__all__ = [
    # Subpackages
    "preprocessing",
    "training",
    "inference",
    "evaluation",
    # Preprocessing
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
    # Training
    "FloodRiskTrainer",
    "SpatialZoneSplitter",
    "TemporalSplitter",
    "IsotonicProbabilityCalibrator",
    "calculate_ece",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_MODEL_FILENAME",
    # Inference
    "RoadFloodRiskScorer",
    "DrainWatchAdjuster",
    "ConformalRiskPredictor",
    "score_to_band",
    "compute_vehicle_risk_scores",
    "RISK_BAND_NAMES",
    # Fusion & Explanation
    "DrainWatchFusionEngine",
    "get_fusion_engine",
    "fuse_drainwatch_risk",
    "RiskExplanationGenerator",
    "RoadRiskExplanation",
    "RiskFactor",
    # Evaluation
    "MetricSummary",
    "find_optimal_threshold_fbeta",
    "calculate_metrics",
    "EvaluationReporter",
    # Legacy placeholders
    "RainfallService",
    "FeatureExtractor",
    "RoadFloodRiskModel",
]
