"""
DrainWatch-to-Flood-Risk Fusion Architecture Module.
Implements the approved Stage 1 + Stage 2 fusion methodology:
- Baseline Preservation: Base GIS/terrain/rainfall model scores all roads (Stage 1).
- Optional CCTV: Missing CCTV observations never break the model (delta = 0.0).
- Confidence & Staleness: Vision evidence is confidence-weighted and temporally decayed.
- Zero Double-Counting: Observational CCTV indicators (blockage, overflow, rise rate)
  only enter Stage 2 and never re-inject static GIS/rainfall features.
- Bounded 0-100 Score: Produces calibrated final road accessibility risk scores in [0, 100].
- Structured Explanations: Every road (and especially high-risk roads) generates
  a structured 'WHY IS THIS ROAD HIGH RISK?' explanation.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

from config.constants import RISK_CLASSES
from ml.drainwatch_adjustment import DrainWatchAdjuster, get_staleness_multiplier, CLASS_DELTA_POINTS
from ml.explanation import RiskExplanationGenerator, RoadRiskExplanation, RiskFactor
from ml.inference.uncertainty import score_to_band


class DrainWatchFusionEngine:
    """
    Core fusion engine combining baseline GIS/hydrological/rainfall risk scores
    with real-time DrainWatch CCTV camera evidence.
    """

    def __init__(
        self,
        adjuster: Optional[DrainWatchAdjuster] = None,
        explanation_generator: Optional[RiskExplanationGenerator] = None,
    ) -> None:
        self.adjuster = adjuster or DrainWatchAdjuster()
        self.explanation_generator = explanation_generator or RiskExplanationGenerator()

    def fuse_single_road(
        self,
        base_risk_score: float,
        drainwatch_features: Optional[Dict[str, Any]] = None,
        road_attributes: Optional[Dict[str, Any]] = None,
        road_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fuse baseline risk score with optional DrainWatch CCTV features for a single road.

        Parameters:
            base_risk_score: Baseline GIS/rainfall risk score (0.0 to 100.0).
            drainwatch_features: Optional dict containing vision metrics:
                - drainwatch_available: bool
                - drainwatch_observation_class: str
                - drainwatch_confidence: float
                - drainwatch_data_age_hours: float
                - drainwatch_blockage_pct: float
                - drainwatch_water_coverage_pct: float
                - drainwatch_overflow_detected: bool
                - drainwatch_water_rise_rate: float
            road_attributes: Optional dict containing terrain, rainfall, and historical context.
            road_id: Optional identifier for the road segment.

        Returns:
            Dict[str, Any] containing:
                - road_id
                - base_risk_score
                - drainwatch_delta
                - flood_access_risk_score
                - risk_class
                - cctv_applied
                - cctv_confidence
                - risk_factors
                - explanation (structured dict)
                - formatted_explanation
        """
        # 1. Check CCTV availability
        dw = drainwatch_features or {}
        is_avail = bool(dw.get("drainwatch_available", False))

        # 2. Compute Stage 2 delta
        if is_avail:
            delta = self.adjuster.compute_single_delta(
                observation_class=dw.get("drainwatch_observation_class"),
                confidence=dw.get("drainwatch_confidence", 1.0),
                age_hours=dw.get("drainwatch_data_age_hours", 0.0),
                drainwatch_score=dw.get("drainwatch_score"),
                blockage_pct=dw.get("drainwatch_blockage_pct"),
                water_coverage_pct=dw.get("drainwatch_water_coverage_pct"),
                overflow_detected=dw.get("drainwatch_overflow_detected"),
                water_rise_rate=dw.get("drainwatch_water_rise_rate"),
            )
            cctv_applied = True
            cctv_conf = dw.get("drainwatch_confidence")
        else:
            # Baseline GIS/rainfall flood risk is strictly preserved
            delta = 0.0
            cctv_applied = False
            cctv_conf = None

        # 3. Final bounded score 0-100
        final_score = float(np.clip(base_risk_score + delta, 0.0, 100.0))
        final_score_rounded = round(final_score, 1)

        # 4. Standard risk class: LOW, MODERATE, HIGH, CRITICAL
        risk_class = score_to_band(final_score_rounded)

        # 5. Assemble context for explanation generation (guarantees zero double-counting)
        combined_data: Dict[str, Any] = dict(road_attributes or {})
        combined_data.update(dw)
        if road_id:
            combined_data["road_id"] = road_id
        combined_data["base_risk_score"] = base_risk_score
        combined_data["drainwatch_delta"] = round(delta, 2)
        combined_data["flood_access_risk_score"] = final_score_rounded
        combined_data["risk_class"] = risk_class

        # 6. Generate structured explanation
        explanation = self.explanation_generator.explain_road(
            road_data=combined_data,
            risk_score=final_score_rounded,
            risk_class=risk_class,
        )

        r_id = explanation.road_id

        return {
            "road_id": r_id,
            "base_risk_score": round(float(base_risk_score), 1),
            "drainwatch_delta": round(float(delta), 2),
            "flood_access_risk_score": final_score_rounded,
            "risk_class": risk_class,
            "is_high_risk": explanation.is_high_risk,
            "cctv_applied": cctv_applied,
            "cctv_confidence": round(float(cctv_conf), 3) if cctv_conf is not None else None,
            "risk_factors": list(explanation.factors),
            "explanation": explanation.to_dict(),
            "formatted_explanation": explanation.format_text(),
        }

    def fuse_dataframe(
        self,
        df: pd.DataFrame,
        base_score_col: str = "base_risk_score",
        output_score_col: str = "flood_access_risk_score",
        delta_col: str = "drainwatch_delta",
        generate_explanations: bool = True,
    ) -> pd.DataFrame:
        """
        Batch-fuses road risk scores across a DataFrame with CCTV features.

        Parameters:
            df: Input DataFrame with base risk scores and optional DrainWatch columns.
            base_score_col: Column name containing baseline risk score.
            output_score_col: Target column for final fused score.
            delta_col: Column name to store CCTV additive delta.
            generate_explanations: If True, attaches structured explanations to each row.

        Returns:
            pd.DataFrame enriched with fused scores, risk class, and explanations.
        """
        if df is None or df.empty:
            return df

        # Step 1: Apply Stage 2 CCTV adjustments
        adjusted_df = self.adjuster.apply_adjustments(
            df=df,
            base_score_col=base_score_col,
            output_col=output_score_col,
            delta_col=delta_col,
        )

        # Step 2: Assign standard risk classes
        final_scores = adjusted_df[output_score_col].values
        risk_classes = [score_to_band(s) for s in final_scores]
        adjusted_df["risk_class"] = risk_classes

        # Step 3: Optional structured explanations
        if generate_explanations:
            explanations = []
            factors_list = []
            for _, row in adjusted_df.iterrows():
                exp = self.explanation_generator.explain_road(
                    road_data=row,
                    risk_score=row[output_score_col],
                    risk_class=row["risk_class"],
                )
                explanations.append(exp.to_dict())
                factors_list.append(exp.factors)

            adjusted_df["explanation"] = explanations
            adjusted_df["risk_factors"] = factors_list

        return adjusted_df


# Module-level convenience functions
_default_engine: Optional[DrainWatchFusionEngine] = None


def get_fusion_engine() -> DrainWatchFusionEngine:
    """Retrieve or initialize singleton fusion engine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = DrainWatchFusionEngine()
    return _default_engine


def fuse_drainwatch_risk(
    base_risk_score: float,
    drainwatch_features: Optional[Dict[str, Any]] = None,
    road_attributes: Optional[Dict[str, Any]] = None,
    road_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to fuse baseline risk with optional DrainWatch CCTV features.
    """
    engine = get_fusion_engine()
    return engine.fuse_single_road(
        base_risk_score=base_risk_score,
        drainwatch_features=drainwatch_features,
        road_attributes=road_attributes,
        road_id=road_id,
    )
