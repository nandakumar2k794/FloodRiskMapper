"""
Risk Explanation Generation Module for Chennai Flood Access Risk Mapper.
Generates structured 'WHY IS THIS ROAD HIGH RISK?' explanations for emergency responders,
urban planners, and routing engines.

Key requirements:
- CCTV observations are optional — missing CCTV must not break explanation generation.
- No double-counting: factors are cleanly mapped to their originating physical category:
  * RAINFALL: IMD thresholds (1h, 3h, 24h accumulation)
  * TERRAIN: Elevation, slope, local depression context, HAND
  * HISTORICAL: Past observed flood frequency and inundation presence
  * DRAINWATCH_CV: Real-time camera evidence (blockage, overflow, water rise rate)
  * HYDROLOGY: Proximity to canals, lakes, rivers, or open drains
- High-risk roads produce structured factor lists with exact names:
  * Heavy rainfall
  * Low elevation
  * Historical vulnerability
  * Drain inlet blockage
  * Increasing water coverage
"""
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd

from config.constants import (
    RISK_CLASSES,
    RAINFALL_THRESHOLDS_MM,
    HAND_FLOOD_THRESHOLD_M,
    DRAINWATCH_CLASS_SCORES,
)


@dataclass
class RiskFactor:
    """
    Granular risk factor contributing to a road segment's flood access risk score.
    """
    factor_name: str          # Standard factor label (e.g., "Heavy rainfall")
    category: str             # "RAINFALL", "TERRAIN", "HISTORICAL", "DRAINWATCH_CV", "HYDROLOGY"
    severity: str             # "MODERATE", "HIGH", "CRITICAL"
    contributing_value: str   # Human-readable value, e.g. "120.0 mm / 24h"
    weight_or_impact: float   # Relative impact on score (0.0 to 1.0 or points)
    source_type: str          # "REAL", "SIMULATED", "INFERRED", "OBSERVED_CV"
    description: str          # Contextual sentence explaining why this increases risk

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RoadRiskExplanation:
    """
    Structured explanation for a road segment's flood risk assessment.
    For roads with high risk (score >= 50.0), provides the 'WHY IS THIS ROAD HIGH RISK?' breakdown.
    """
    road_id: str
    headline: str                       # "WHY IS THIS ROAD HIGH RISK?"
    risk_score: float                   # 0.0 to 100.0
    risk_class: str                     # "LOW", "MODERATE", "HIGH", "CRITICAL"
    is_high_risk: bool                  # True if score >= 50.0 (HIGH or CRITICAL)
    factors: List[str]                  # Simple list of active factor names
    structured_factors: List[RiskFactor]# Detailed factor dataclasses
    summary_text: str                   # Plain-text narrative
    cctv_available: bool                # Whether CCTV camera coverage was active
    cctv_confidence: Optional[float]    # Detection confidence if CCTV available
    data_sources: List[str]             # List of data sources utilized

    def to_dict(self) -> Dict[str, Any]:
        return {
            "road_id": self.road_id,
            "headline": self.headline,
            "risk_score": round(self.risk_score, 1),
            "risk_class": self.risk_class,
            "is_high_risk": self.is_high_risk,
            "factors": list(self.factors),
            "structured_factors": [f.to_dict() for f in self.structured_factors],
            "summary_text": self.summary_text,
            "cctv_available": self.cctv_available,
            "cctv_confidence": round(self.cctv_confidence, 3) if self.cctv_confidence is not None else None,
            "data_sources": list(self.data_sources),
        }

    def format_text(self) -> str:
        """Render a readable terminal/markdown block."""
        lines = [
            f"=== {self.headline} ===",
            f"Road ID: {self.road_id} | Score: {self.risk_score:.1f}/100 ({self.risk_class})",
            f"CCTV Active: {'Yes' if self.cctv_available else 'No'}",
            "",
            "Key Contributing Factors:",
        ]
        if not self.factors:
            lines.append("  (No elevated risk factors detected)")
        else:
            for f in self.structured_factors:
                lines.append(f"  • {f.factor_name}: {f.contributing_value} [{f.severity}]")
        lines.append("")
        lines.append(f"Summary: {self.summary_text}")
        lines.append(f"Sources: {', '.join(self.data_sources)}")
        return "\n".join(lines)


class RiskExplanationGenerator:
    """
    Engine to identify, rank, and explain drivers of road flood risk.
    Guarantees no double-counting and seamless operation when CCTV is missing.
    """

    # Physical thresholds
    RAINFALL_HEAVY_24H_MM = RAINFALL_THRESHOLDS_MM.get("heavy", 64.5)
    RAINFALL_VERY_HEAVY_24H_MM = RAINFALL_THRESHOLDS_MM.get("very_heavy", 115.6)
    RAINFALL_EXTREME_24H_MM = RAINFALL_THRESHOLDS_MM.get("extremely_heavy", 204.5)
    RAINFALL_HOURLY_MODERATE_MM = 15.6
    RAINFALL_HOURLY_HEAVY_MM = 35.0

    ELEVATION_LOW_M = 4.0
    LOCAL_DEPRESSION_THRESHOLD_M = 0.0
    SLOPE_FLAT_DEG = 1.0

    HISTORICAL_COUNT_THRESHOLD = 1.0
    HISTORICAL_FREQ_THRESHOLD = 0.20

    BLOCKAGE_THRESHOLD_PCT = 25.0
    WATER_RISE_RATE_THRESHOLD = 0.50
    MIN_CCTV_CONFIDENCE = 0.25
    MAX_CCTV_STALENESS_HOURS = 12.0

    def explain_road(
        self,
        road_data: Union[pd.Series, Dict[str, Any]],
        risk_score: Optional[float] = None,
        risk_class: Optional[str] = None,
    ) -> RoadRiskExplanation:
        """
        Generate structured explanation for a single road segment.
        """
        # Convert Series/dict to safe getter
        def _get(key: str, default: Any = None) -> Any:
            if isinstance(road_data, pd.Series):
                val = road_data.get(key, default)
            elif isinstance(road_data, dict):
                val = road_data.get(key, default)
            else:
                val = getattr(road_data, key, default)
            if val is None or pd.isna(val):
                return default
            return val

        road_id = str(_get("road_id", _get("segment_id", "unknown_road")))

        # Determine risk score and class
        final_score = risk_score
        if final_score is None:
            final_score = _get("flood_access_risk_score", _get("adjusted_risk_score", _get("base_risk_score", 0.0)))
        final_score = float(final_score)

        final_class = risk_class
        if final_class is None:
            final_class = _get("risk_class")
            if not final_class:
                if final_score >= 75.0:
                    final_class = "CRITICAL"
                elif final_score >= 50.0:
                    final_class = "HIGH"
                elif final_score >= 25.0:
                    final_class = "MODERATE"
                else:
                    final_class = "LOW"
        final_class = str(final_class).upper()

        is_high_risk = final_score >= 50.0 or final_class in ["HIGH", "CRITICAL"]

        structured_factors: List[RiskFactor] = []
        data_sources: List[str] = ["SRTM_DEM", "OSM_ROADS"]

        # -------------------------------------------------------------
        # 1. RAINFALL FACTOR: "Heavy rainfall"
        # -------------------------------------------------------------
        r24 = float(_get("rainfall_24h", 0.0))
        r1 = float(_get("rainfall_1h", 0.0))
        r3 = float(_get("rainfall_3h", 0.0))
        r_source = str(_get("rainfall_source_type", "SIMULATED")).upper()
        if r24 > 0.0 or r1 > 0.0:
            data_sources.append(f"RAINFALL_{r_source}")

        if r24 >= self.RAINFALL_HEAVY_24H_MM or r1 >= self.RAINFALL_HOURLY_MODERATE_MM or r3 >= 35.0:
            if r24 >= self.RAINFALL_EXTREME_24H_MM or r1 >= 50.0:
                rf_sev = "CRITICAL"
                rf_impact = 0.35
            elif r24 >= self.RAINFALL_VERY_HEAVY_24H_MM or r1 >= self.RAINFALL_HOURLY_HEAVY_MM:
                rf_sev = "HIGH"
                rf_impact = 0.30
            else:
                rf_sev = "MODERATE"
                rf_impact = 0.20

            detail_str = f"{r24:.1f} mm / 24h"
            if r1 > 0.0:
                detail_str += f" ({r1:.1f} mm/h)"

            structured_factors.append(
                RiskFactor(
                    factor_name="Heavy rainfall",
                    category="RAINFALL",
                    severity=rf_sev,
                    contributing_value=detail_str,
                    weight_or_impact=rf_impact,
                    source_type=r_source,
                    description=(
                        f"Cumulative rainfall ({detail_str}) meets or exceeds the IMD heavy rainfall "
                        f"threshold ({self.RAINFALL_HEAVY_24H_MM} mm), causing severe overland runoff."
                    ),
                )
            )

        # -------------------------------------------------------------
        # 2. TERRAIN FACTOR: "Low elevation"
        # -------------------------------------------------------------
        elev_mean = float(_get("elevation_mean", 10.0))
        loc_ctx = float(_get("local_elevation_context", 0.0))
        slope = float(_get("slope", 2.0))
        hand = float(_get("hand_value", 5.0))

        is_low_elevation = (
            loc_ctx <= self.LOCAL_DEPRESSION_THRESHOLD_M
            or elev_mean <= self.ELEVATION_LOW_M
            or hand <= HAND_FLOOD_THRESHOLD_M
            or slope <= self.SLOPE_FLAT_DEG
        )

        if is_low_elevation:
            if loc_ctx <= -2.0 or elev_mean <= 2.0:
                t_sev = "CRITICAL"
                t_impact = 0.25
            elif loc_ctx <= -0.5 or elev_mean <= self.ELEVATION_LOW_M:
                t_sev = "HIGH"
                t_impact = 0.20
            else:
                t_sev = "MODERATE"
                t_impact = 0.15

            t_val_parts = []
            if loc_ctx <= 0.0:
                t_val_parts.append(f"local depression {loc_ctx:+.1f}m")
            if elev_mean <= 6.0:
                t_val_parts.append(f"mean elev {elev_mean:.1f}m")
            if slope <= self.SLOPE_FLAT_DEG:
                t_val_parts.append(f"slope {slope:.1f}°")
            t_val_str = ", ".join(t_val_parts) if t_val_parts else f"mean elev {elev_mean:.1f}m"

            structured_factors.append(
                RiskFactor(
                    factor_name="Low elevation",
                    category="TERRAIN",
                    severity=t_sev,
                    contributing_value=t_val_str,
                    weight_or_impact=t_impact,
                    source_type="REAL",
                    description=(
                        f"Road sits in low-lying topography ({t_val_str}), making it a natural accumulation "
                        "point for gravity-fed stormwater runoff."
                    ),
                )
            )

        # -------------------------------------------------------------
        # 3. HISTORICAL FACTOR: "Historical vulnerability"
        # -------------------------------------------------------------
        hist_count = float(_get("historical_flood_count", 0.0))
        hist_freq = float(_get("historical_flood_frequency", 0.0))
        hist_pres = float(_get("historical_inundation_presence", 0.0))
        hist_source = str(_get("primary_evidence_type", "INFERRED")).upper()
        if hist_count > 0 or hist_freq > 0:
            data_sources.append("HISTORICAL_RECORDS")

        if hist_count >= self.HISTORICAL_COUNT_THRESHOLD or hist_freq >= self.HISTORICAL_FREQ_THRESHOLD or hist_pres > 0.0:
            if hist_count >= 3.0 or hist_freq >= 0.50:
                h_sev = "CRITICAL"
                h_impact = 0.25
            elif hist_count >= 2.0 or hist_freq >= 0.30:
                h_sev = "HIGH"
                h_impact = 0.20
            else:
                h_sev = "MODERATE"
                h_impact = 0.15

            h_val = f"{int(hist_count)} prior inundation events (frequency: {hist_freq:.2f})"
            structured_factors.append(
                RiskFactor(
                    factor_name="Historical vulnerability",
                    category="HISTORICAL",
                    severity=h_sev,
                    contributing_value=h_val,
                    weight_or_impact=h_impact,
                    source_type=hist_source,
                    description=(
                        f"Segment has documented repeat inundation history ({h_val}), indicating chronic "
                        "drainage insufficiency during monsoon events."
                    ),
                )
            )

        # -------------------------------------------------------------
        # 4 & 5. DRAINWATCH CCTV FACTORS (Optional - only when active)
        # -------------------------------------------------------------
        is_cctv_avail = bool(_get("drainwatch_available", False))
        cctv_conf_raw = _get("drainwatch_confidence", None)
        cctv_conf = float(cctv_conf_raw) if cctv_conf_raw is not None else None
        cctv_age = float(_get("drainwatch_data_age_hours", 0.0))
        obs_class = _get("drainwatch_observation_class", None)
        if obs_class is not None:
            obs_class = str(obs_class).strip().lower()

        blockage_pct = float(_get("drainwatch_blockage_pct", 0.0))
        water_cov_pct = float(_get("drainwatch_water_coverage_pct", 0.0))
        water_rise_rate = float(_get("drainwatch_water_rise_rate", 0.0))
        overflow_flag = bool(_get("drainwatch_overflow_detected", False))

        cctv_valid = (
            is_cctv_avail
            and (cctv_conf is None or cctv_conf >= self.MIN_CCTV_CONFIDENCE)
            and (cctv_age <= self.MAX_CCTV_STALENESS_HOURS)
        )

        if cctv_valid:
            data_sources.append("DRAINWATCH_CCTV")
            conf_str = f"conf: {cctv_conf:.2f}" if cctv_conf is not None else "verified"

            # 4. Drain inlet blockage
            is_blocked = (
                blockage_pct >= self.BLOCKAGE_THRESHOLD_PCT
                or obs_class in ["drain_blocked_garbage", "drain_blocked_silt"]
            )
            if is_blocked:
                b_sev = "CRITICAL" if blockage_pct >= 60.0 else "HIGH" if blockage_pct >= 40.0 else "MODERATE"
                b_val = f"{blockage_pct:.1f}% blockage" if blockage_pct > 0 else f"blocked ({obs_class})"
                structured_factors.append(
                    RiskFactor(
                        factor_name="Drain inlet blockage",
                        category="DRAINWATCH_CV",
                        severity=b_sev,
                        contributing_value=f"{b_val} ({conf_str})",
                        weight_or_impact=0.20,
                        source_type="OBSERVED_CV",
                        description=(
                            f"CCTV camera verified physical drain inlet obstruction ({b_val}), "
                            "severely reducing hydraulic discharge capacity."
                        ),
                    )
                )

            # 5. Increasing water coverage
            is_rising = (
                water_rise_rate >= self.WATER_RISE_RATE_THRESHOLD
                or (water_cov_pct >= 25.0 and water_rise_rate > 0.0)
            )
            if is_rising:
                r_sev = "CRITICAL" if water_rise_rate >= 2.5 else "HIGH" if water_rise_rate >= 1.0 else "MODERATE"
                r_val = f"+{water_rise_rate:.2f}%/min (coverage: {water_cov_pct:.1f}%)"
                structured_factors.append(
                    RiskFactor(
                        factor_name="Increasing water coverage",
                        category="DRAINWATCH_CV",
                        severity=r_sev,
                        contributing_value=f"{r_val} ({conf_str})",
                        weight_or_impact=0.25,
                        source_type="OBSERVED_CV",
                        description=(
                            f"Real-time video temporal analysis tracks positive water velocity ({r_val}), "
                            "confirming ongoing water accumulation."
                        ),
                    )
                )

            # Additional CCTV evidence: Drain overflow & Road waterlogging
            if overflow_flag or obs_class == "drain_overflowing":
                structured_factors.append(
                    RiskFactor(
                        factor_name="Drain overflow",
                        category="DRAINWATCH_CV",
                        severity="CRITICAL",
                        contributing_value=f"Active overflow detected ({conf_str})",
                        weight_or_impact=0.30,
                        source_type="OBSERVED_CV",
                        description="Stormwater is surging out of drain channels directly onto the road surface.",
                    )
                )

            if obs_class == "road_waterlogged" or water_cov_pct >= 60.0:
                structured_factors.append(
                    RiskFactor(
                        factor_name="Road waterlogging",
                        category="DRAINWATCH_CV",
                        severity="CRITICAL",
                        contributing_value=f"{water_cov_pct:.1f}% road surface submerged",
                        weight_or_impact=0.30,
                        source_type="OBSERVED_CV",
                        description="Significant portion of the carriageway is already submerged beneath standing water.",
                    )
                )

        # -------------------------------------------------------------
        # 6. HYDROLOGY FACTOR: "Proximity to water body"
        # -------------------------------------------------------------
        dist_water = float(_get("dist_to_water_m", 5000.0))
        near_water_50 = float(_get("near_water_50m", 0.0))
        crosses_water = float(_get("crosses_water", 0.0))

        if near_water_50 > 0.0 or dist_water <= 100.0 or crosses_water > 0.0:
            data_sources.append("HYDROLOGY_NETWORK")
            hw_sev = "CRITICAL" if dist_water <= 30.0 or crosses_water > 0.0 else "HIGH"
            hw_val = f"{dist_water:.0f}m from canal/river" if crosses_water == 0.0 else "crosses water channel"
            structured_factors.append(
                RiskFactor(
                    factor_name="Proximity to water body",
                    category="HYDROLOGY",
                    severity=hw_sev,
                    contributing_value=hw_val,
                    weight_or_impact=0.15,
                    source_type="REAL",
                    description=(
                        f"Road is immediately adjacent to surface water ({hw_val}), subject to backwater effect "
                        "and bank overtopping during high tide or surge."
                    ),
                )
            )

        # -------------------------------------------------------------
        # Assemble Factors and Summary
        # -------------------------------------------------------------
        # Priority order of factor names:
        # 1. Heavy rainfall
        # 2. Low elevation
        # 3. Historical vulnerability
        # 4. Drain inlet blockage
        # 5. Increasing water coverage
        # followed by other active factors
        PRIORITY_FACTOR_ORDER = [
            "Heavy rainfall",
            "Low elevation",
            "Historical vulnerability",
            "Drain inlet blockage",
            "Increasing water coverage",
            "Drain overflow",
            "Road waterlogging",
            "Proximity to water body",
        ]

        # Sort structured factors according to preferred order, then by severity
        severity_rank = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}

        def _sort_key(f: RiskFactor) -> Tuple[int, int]:
            order_idx = PRIORITY_FACTOR_ORDER.index(f.factor_name) if f.factor_name in PRIORITY_FACTOR_ORDER else 99
            sev_idx = severity_rank.get(f.severity, 4)
            return (order_idx, sev_idx)

        structured_factors.sort(key=_sort_key)
        factor_names = [f.factor_name for f in structured_factors]

        # Headline
        headline = "WHY IS THIS ROAD HIGH RISK?" if is_high_risk else "ROAD FLOOD RISK ASSESSMENT"

        # Summary text
        if is_high_risk:
            if factor_names:
                summary_factors = ", ".join(f"{f.factor_name} ({f.contributing_value})" for f in structured_factors[:4])
                summary_text = (
                    f"Road {road_id} is evaluated at {final_class} risk ({final_score:.1f}/100). "
                    f"Primary risk drivers: {summary_factors}."
                )
            else:
                summary_text = (
                    f"Road {road_id} is evaluated at {final_class} risk ({final_score:.1f}/100) "
                    "due to cumulative network proximity and base hazard indicators."
                )
        else:
            summary_text = (
                f"Road {road_id} is currently assessed at {final_class} risk ({final_score:.1f}/100). "
                "Conditions remain within manageable operational thresholds."
            )

        return RoadRiskExplanation(
            road_id=road_id,
            headline=headline,
            risk_score=final_score,
            risk_class=final_class,
            is_high_risk=is_high_risk,
            factors=factor_names,
            structured_factors=structured_factors,
            summary_text=summary_text,
            cctv_available=cctv_valid,
            cctv_confidence=cctv_conf if cctv_valid else None,
            data_sources=sorted(list(set(data_sources))),
        )

    def explain_dataframe(
        self,
        df: pd.DataFrame,
        high_risk_only: bool = False,
    ) -> List[RoadRiskExplanation]:
        """
        Batch-generate explanations for all or high-risk road rows in a DataFrame.
        """
        explanations: List[RoadRiskExplanation] = []
        for _, row in df.iterrows():
            exp = self.explain_road(row)
            if not high_risk_only or exp.is_high_risk:
                explanations.append(exp)
        return explanations
