"""
XGBoost Primary Training Pipeline with Monotonic Constraints and Zone Cross-Validation.
Saves and loads trained models to models/ directory.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from ml.preprocessing.weights import calculate_scale_pos_weight, compute_sample_weights
from ml.training.calibration import IsotonicProbabilityCalibrator, calculate_ece
from ml.training.split import SpatialZoneSplitter


DEFAULT_MODEL_DIR = Path("models")
DEFAULT_MODEL_FILENAME = "flood_risk_xgboost.pkl"

MONOTONIC_INCREASING_FEATURES = [
    "rainfall_15min",
    "rainfall_1h",
    "rainfall_3h",
    "rainfall_6h",
    "rainfall_24h",
    "historical_flood_frequency",
    "hist_freq_x_rainfall_24h",
]


class FloodRiskTrainer:
    """
    XGBoost primary model trainer with monotonic rainfall constraints,
    spatial GroupKFold cross-validation, and calibrated risk probability output.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        use_monotonic_constraints: bool = True,
        use_ensemble: bool = False,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.use_monotonic_constraints = use_monotonic_constraints
        self.use_ensemble = use_ensemble

        self.primary_model: Optional[xgb.XGBClassifier] = None
        self.secondary_model: Optional[Any] = None
        self.calibrator: Optional[IsotonicProbabilityCalibrator] = None
        self.feature_names: List[str] = []
        self.training_metadata: Dict[str, Any] = {}

    def _build_monotonic_constraints(self, feature_names: List[str]) -> Tuple[int, ...]:
        """Generate monotonic constraint tuple (+1, 0, or -1)."""
        constraints = []
        for feat in feature_names:
            if feat in MONOTONIC_INCREASING_FEATURES:
                constraints.append(1)
            else:
                constraints.append(0)
        return tuple(constraints)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        scale_pos_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Train XGBoost with monotonic constraints and fit probability calibrator on validation zone.
        """
        self.feature_names = list(X_train.columns)
        calc_scale = (
            scale_pos_weight
            if scale_pos_weight is not None
            else calculate_scale_pos_weight(y_train)
        )

        xgb_params: Dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "scale_pos_weight": calc_scale,
            "random_state": self.random_state,
            "eval_metric": "logloss",
            "n_jobs": 1,
        }

        if self.use_monotonic_constraints:
            xgb_params["monotone_constraints"] = self._build_monotonic_constraints(self.feature_names)

        self.primary_model = xgb.XGBClassifier(**xgb_params)
        self.primary_model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            verbose=False,
        )

        if self.use_ensemble:
            try:
                import lightgbm as lgb
                self.secondary_model = lgb.LGBMClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                    verbose=-1,
                )
                self.secondary_model.fit(X_train, y_train, sample_weight=sample_weights)
            except Exception:
                X_imp = X_train.fillna(0.0)
                self.secondary_model = RandomForestClassifier(
                    n_estimators=min(100, self.n_estimators),
                    max_depth=self.max_depth + 2,
                    random_state=self.random_state,
                    n_jobs=1,
                )
                self.secondary_model.fit(X_imp, y_train, sample_weight=sample_weights)

        # Fit Calibrator
        self.calibrator = IsotonicProbabilityCalibrator()
        report: Dict[str, Any] = {
            "n_train": len(X_train),
            "scale_pos_weight": calc_scale,
            "features": self.feature_names,
        }

        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_raw_probs = self.predict_proba_uncalibrated(X_val)
            self.calibrator.fit(val_raw_probs, y_val)
            val_cal_probs = self.calibrator.calibrate(val_raw_probs)

            valid_mask = ~np.isnan(y_val)
            if np.sum(valid_mask) > 0 and len(np.unique(y_val[valid_mask])) > 1:
                report["val_auroc"] = float(roc_auc_score(y_val[valid_mask], val_cal_probs[valid_mask]))
                report["val_auprc"] = float(average_precision_score(y_val[valid_mask], val_cal_probs[valid_mask]))
            report["val_ece"] = calculate_ece(y_val, val_cal_probs)
        else:
            train_raw_probs = self.predict_proba_uncalibrated(X_train)
            self.calibrator.fit(train_raw_probs, y_train, sample_weights=sample_weights)

        self.training_metadata = report
        return report

    def predict_proba_uncalibrated(self, X: pd.DataFrame) -> np.ndarray:
        """Raw uncalibrated predicted probabilities."""
        if self.primary_model is None:
            raise RuntimeError("Primary model is not trained.")

        X_aligned = X[self.feature_names]
        p_primary = self.primary_model.predict_proba(X_aligned)[:, 1]

        if self.secondary_model is not None:
            try:
                X_imp = X_aligned.fillna(0.0)
                p_secondary = self.secondary_model.predict_proba(X_imp)[:, 1]
                return 0.6 * p_primary + 0.4 * p_secondary
            except Exception:
                return p_primary

        return p_primary

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated predicted probabilities in [0.0, 1.0]."""
        raw = self.predict_proba_uncalibrated(X)
        if self.calibrator is not None and self.calibrator.is_fitted:
            return self.calibrator.calibrate(raw)
        return raw

    def spatial_cross_validate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        zones: np.ndarray,
        n_splits: int = 3,
    ) -> List[Dict[str, float]]:
        """
        Spatial GroupKFold cross-validation grouped by zone_no.
        """
        unique_zones = len(np.unique(zones))
        actual_splits = min(n_splits, unique_zones)
        if actual_splits < 2:
            return []

        gkf = GroupKFold(n_splits=actual_splits)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=zones)):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_v, y_v = X.iloc[val_idx], y[val_idx]

            fold_trainer = FloodRiskTrainer(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state + fold,
                use_monotonic_constraints=self.use_monotonic_constraints,
                use_ensemble=False,
            )
            fold_trainer.train(X_tr, y_tr)
            probs = fold_trainer.predict_proba(X_v)

            valid_mask = ~np.isnan(y_v)
            if len(np.unique(y_v[valid_mask])) > 1:
                auroc = float(roc_auc_score(y_v[valid_mask], probs[valid_mask]))
                auprc = float(average_precision_score(y_v[valid_mask], probs[valid_mask]))
            else:
                auroc = 0.5
                auprc = 0.5

            ece = calculate_ece(y_v, probs)

            fold_results.append({
                "fold": fold + 1,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "val_auroc": auroc,
                "val_auprc": auprc,
                "val_ece": ece,
            })

        return fold_results

    def save(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Serialize model bundle to disk under models/ directory.
        """
        if output_path is None:
            path = DEFAULT_MODEL_DIR / DEFAULT_MODEL_FILENAME
        else:
            path = Path(output_path)

        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "primary_model": self.primary_model,
            "secondary_model": self.secondary_model,
            "calibrator": self.calibrator,
            "feature_names": self.feature_names,
            "training_metadata": self.training_metadata,
        }
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        return path

    def load(self, model_path: Optional[Union[str, Path]] = None) -> "FloodRiskTrainer":
        """
        Deserialize model bundle from disk.
        """
        if model_path is None:
            path = DEFAULT_MODEL_DIR / DEFAULT_MODEL_FILENAME
        else:
            path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found at: {path}")

        with open(path, "rb") as f:
            bundle = pickle.load(f)

        self.primary_model = bundle.get("primary_model")
        self.secondary_model = bundle.get("secondary_model")
        self.calibrator = bundle.get("calibrator")
        self.feature_names = bundle.get("feature_names", [])
        self.training_metadata = bundle.get("training_metadata", {})
        return self
