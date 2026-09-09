"""
End-to-End Road Flood Access Risk Scorer Module.
Implements §11, §14, and §15 of the ML methodology:
- Predicts calibrated accessibility probabilities via Stage 1 model.
- Scales probability to base 0-100 risk score.
- Applies Stage 2 DrainWatch additive evidence deltas for CCTV-covered roads.
- Evaluates scenario extrapolation bounds (> 300mm rainfall).
- Enriches output with conformal uncertainty prediction sets and confidence flags.
- Computes vehicle-specific modulated accessibility risk scores.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

from config.constants import RISK_CLASSES, DEPTH_THRESHOLDS_CM
from ml.feature_matrix import FeatureMatrixBuilder, MODEL_FEATURE_NAMES
from ml.trainer import FloodRiskTrainer
from ml.drainwatch_adjustment import DrainWatchAdjuster
from ml.inference.uncertainty import ConformalRiskPredictor, score_to_band
from ml.explanation import RiskExplanationGenerator


# Vehicle depth modulation points (offsets from base risk score)
VEHICLE_RISK_OFFSETS = {
    "motorcycle": {"LOW": 5.0, "MEDIUM": 10.0, "HIGH": 12.0, "CRITICAL": 10.0},
    "car": {"LOW": 0.0, "MEDIUM": 0.0, "HIGH": 0.0, "CRITICAL": 0.0},
    "suv": {"LOW": -2.0, "MEDIUM": -3.0, "HIGH": -3.0, "CRITICAL": -2.0},
    "bus": {"LOW": -5.0, "MEDIUM": -5.0, "HIGH": -4.0, "CRITICAL": -2.0},
    "pedestrian": {"LOW": 10.0, "MEDIUM": 15.0, "HIGH": 15.0, "CRITICAL": 10.0},
}

# Max 24h rainfall observed across Chennai training benchmarks (2015, 2021, 2023)
MAX_TRAINING_RAINFALL_24H_MM = 300.0


def compute_vehicle_risk_scores(base_score: float, risk_band: str) -> Dict[str, float]:
    """Calculate vehicle-specific modulated scores given base score and risk band."""
    scores = {}
    for v_type, band_offsets in VEHICLE_RISK_OFFSETS.items():
        offset = band_offsets.get(risk_band, 0.0)
        v_score = np.clip(base_score + offset, 0.0, 100.0)
        scores[v_type] = round(float(v_score), 1)
    return scores


class RoadFloodRiskScorer:
    """
    End-to-end inference and scoring pipeline.
    """

    def __init__(
        self,
        trainer: Optional[FloodRiskTrainer] = None,
        calibrator: Optional[Any] = None,
        feature_builder: Optional[FeatureMatrixBuilder] = None,
        drainwatch_adjuster: Optional[DrainWatchAdjuster] = None,
        uncertainty_predictor: Optional[ConformalRiskPredictor] = None,
        explanation_generator: Optional[RiskExplanationGenerator] = None,
    ) -> None:
        self.trainer = trainer or FloodRiskTrainer()
        self.feature_builder = feature_builder or FeatureMatrixBuilder()
        self.drainwatch_adjuster = drainwatch_adjuster or DrainWatchAdjuster()
        self.uncertainty_predictor = uncertainty_predictor or ConformalRiskPredictor()
        self.explanation_generator = explanation_generator or RiskExplanationGenerator()

    def score_network(
        self,
        roads_gdf: gpd.GeoDataFrame,
        rainfall_df: Optional[pd.DataFrame] = None,
        drainwatch_df: Optional[pd.DataFrame] = None,
    ) -> gpd.GeoDataFrame:
        """
        Score accessibility risk across all road segments.

        Parameters:
            roads_gdf: Road network GeoDataFrame (with static terrain, hydrology, historical features).
            rainfall_df: Dynamic rainfall windows per road or scenario broadcast.
            drainwatch_df: Vision-derived CCTV drain state records.

        Returns:
            gpd.GeoDataFrame enriched with:
                - 'flood_access_risk_score': Final 0-100 score (calibrated & CCTV-adjusted)
                - 'base_risk_score': Unadjusted Stage 1 score (0-100)
                - 'drainwatch_delta': Additive points from CCTV
                - 'risk_class': LOW, MEDIUM, HIGH, CRITICAL
                - 'vehicle_scores': Dict of vehicle-specific scores
                - 'prediction_set': Conformal risk band coverage set
                - 'is_uncertain': High epistemic uncertainty flag
                - 'scenario_extrapolation': Boolean flag if rainfall exceeds training envelope
        """
        if roads_gdf is None or roads_gdf.empty:
            return roads_gdf

        # 1. Feature matrix assembly
        combined = self.feature_builder.combine_features(
            static_roads_df=roads_gdf,
            rainfall_df=rainfall_df,
            drainwatch_df=drainwatch_df,
            include_drainwatch=True,
        )

        # 2. Check for scenario extrapolation (> 300mm 24h rain)
        max_scenario_rain = float(combined["rainfall_24h"].max()) if "rainfall_24h" in combined.columns else 0.0
        is_extrapolating = max_scenario_rain > MAX_TRAINING_RAINFALL_24H_MM

        # 3. Stage 1 Model Inference
        # Extract features aligned with model training columns
        feat_cols = self.trainer.feature_names if self.trainer.feature_names else [
            c for c in MODEL_FEATURE_NAMES if c in combined.columns
        ]
        
        # Ensure all columns present
        for col in feat_cols:
            if col not in combined.columns:
                combined[col] = 0.0

        X = combined[feat_cols]

        if self.trainer.primary_model is not None:
            calibrated_probs = self.trainer.predict_proba(X)
        else:
            # Fallback heuristic if untrained placeholder
            r24 = combined["rainfall_24h"].fillna(0.0).values
            slope = combined["slope"].fillna(1.0).values
            loc_ctx = combined["local_elevation_context"].fillna(0.0).values
            # Simple physically monotonic heuristic probability
            raw_logits = (r24 / 150.0) - (slope * 0.1) - (loc_ctx * 0.2)
            calibrated_probs = 1.0 / (1.0 + np.exp(-raw_logits))

        combined["calibrated_prob"] = np.round(calibrated_probs, 4)
        base_scores = np.round(calibrated_probs * 100.0, 1)
        combined["base_risk_score"] = base_scores

        # 4. Stage 2 DrainWatch Additive Adjustment
        adjusted_df = self.drainwatch_adjuster.apply_adjustments(
            df=combined,
            base_score_col="base_risk_score",
            output_col="flood_access_risk_score",
            delta_col="drainwatch_delta",
        )

        # 5. Risk band classification
        final_scores = adjusted_df["flood_access_risk_score"].values
        risk_classes = [score_to_band(s) for s in final_scores]
        adjusted_df["risk_class"] = risk_classes

        # 6. Conformal Uncertainty Sets
        enriched_df = self.uncertainty_predictor.enrich_with_uncertainty(
            df=adjusted_df,
            prob_col="calibrated_prob",
            extrapolation=is_extrapolating,
        )

        # 7. Vehicle-Specific Scores
        vehicle_scores_list = []
        for s, band in zip(final_scores, risk_classes):
            v_scores = compute_vehicle_risk_scores(s, band)
            vehicle_scores_list.append(v_scores)
        enriched_df["vehicle_scores"] = vehicle_scores_list

        # 8. Structured Risk Explanations
        explanations = []
        factors_list = []
        for _, row in enriched_df.iterrows():
            exp = self.explanation_generator.explain_road(
                road_data=row,
                risk_score=row["flood_access_risk_score"],
                risk_class=row["risk_class"],
            )
            explanations.append(exp.to_dict())
            factors_list.append(exp.factors)
        enriched_df["explanation"] = explanations
        enriched_df["risk_factors"] = factors_list

        # Convert back to GeoDataFrame if input had geometry
        if "geometry" in roads_gdf.columns and not isinstance(enriched_df, gpd.GeoDataFrame):
            enriched_df = gpd.GeoDataFrame(enriched_df, geometry="geometry", crs=roads_gdf.crs)

        return enriched_df
