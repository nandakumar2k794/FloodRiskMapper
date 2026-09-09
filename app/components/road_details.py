"""
Selected Road Details Panel Component for Streamlit Dashboard.
Displays all required fields for inspecting an individual road segment:
- Road ID
- Flood Risk (0-100 score + risk class badge: LOW, MODERATE, HIGH, CRITICAL)
- Rainfall (24h accumulation & 1h peak mm)
- Elevation (mean metres)
- Slope (degrees)
- Drain Condition (CCTV condition or baseline drainage status)
- Blockage (%)
- Water Coverage (%)
- Historical Risk (prior events & frequency)
- Recommendation (operational action / rerouting guidance)
- Structured 'WHY IS THIS ROAD HIGH RISK?' explanation for elevated segments
"""
from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd


def get_operational_recommendation(
    risk_score: float,
    risk_class: str,
    overflow_detected: bool = False,
    blockage_pct: float = 0.0,
    elev: float = 10.0,
) -> str:
    """Generate defensible emergency and municipal operational recommendations."""
    if risk_score >= 75.0 or overflow_detected:
        if blockage_pct >= 50.0:
            return "🚨 CRITICAL: Road impassable. Dispatch emergency desilting/suction super-suckers immediately. Issue active transit detour for all vehicles."
        elif elev <= 3.5:
            return "🚨 CRITICAL: Severe waterlogging in natural depression. Deploy mobile diesel dewatering pumps. Close road to non-emergency traffic."
        else:
            return "🚨 CRITICAL: Severe flood hazard. Implement immediate police barricades and reroute bus/transit traffic."
    elif risk_score >= 50.0:
        if blockage_pct >= 30.0:
            return "⚠️ HIGH RISK: Storm drain inlet obstruction detected. Deploy GCC rapid response clearing crew. Divert low-clearance vehicles."
        else:
            return "⚠️ HIGH RISK: Heavy runoff accumulation. Restrict two-wheelers and sedans. Monitor water-rise velocity."
    elif risk_score >= 25.0:
        return "🟡 MODERATE: Minor water accumulation possible. Maintain routine stormwater monitoring. Safe for all vehicle types with standard caution."
    else:
        return "🟢 LOW RISK: Normal road accessibility. Clear drainage flow. No emergency intervention required."


def render_selected_road_panel(
    road_data: Dict[str, Any],
    rainfall_data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Render full selected-road details panel with all 10 required fields.
    """
    # Safe field getters
    def _g(k: str, default: Any = None) -> Any:
        v = road_data.get(k, default)
        return default if v is None or pd.isna(v) else v

    road_id = str(_g("segment_id", _g("road_id", "Unknown Road")))
    road_name = str(_g("name", "Chennai Road Segment"))
    highway_class = str(_g("highway", "primary")).capitalize()

    # 1. Flood Risk Score & Class
    risk_score = float(_g("flood_access_risk_score", _g("base_risk_score", 0.0)))
    risk_class = str(_g("risk_class", "LOW")).upper()
    delta = float(_g("drainwatch_delta", 0.0))
    base_score = float(_g("base_risk_score", risk_score))

    # 2. Rainfall
    rain = rainfall_data or {}
    r24 = float(_g("rainfall_24h", rain.get("rainfall_24h", 0.0)))
    r1 = float(_g("rainfall_1h", rain.get("rainfall_1h", 0.0)))

    # 3. Elevation & 4. Slope
    elev = float(_g("elevation_mean", 10.0))
    slope = float(_g("slope", 1.5))
    loc_ctx = float(_g("local_elevation_context", 0.0))

    # 5. Drain Condition, 6. Blockage %, 7. Water Coverage %
    cctv_avail = bool(_g("drainwatch_available", False))
    obs_class_raw = _g("drainwatch_observation_class", None)
    if obs_class_raw:
        drain_condition = str(obs_class_raw).replace("_", " ").title()
    elif _g("has_adjacent_drain", 0.0) > 0:
        drain_condition = "Adjacent Storm Drain (Unmonitored)"
    else:
        drain_condition = "No Immediate Drain Coverage"

    blockage_pct = float(_g("drainwatch_blockage_pct", 0.0))
    water_cov_pct = float(_g("drainwatch_water_coverage_pct", 0.0))
    overflow_detected = bool(_g("drainwatch_overflow_detected", False))
    cctv_conf = _g("drainwatch_confidence", None)

    # 8. Historical Risk
    hist_count = int(_g("historical_flood_count", 0))
    hist_freq = float(_g("historical_flood_frequency", 0.0))
    hist_evidence = str(_g("primary_evidence_type", "INFERRED"))

    # 9. Recommendation
    recommendation = get_operational_recommendation(
        risk_score=risk_score,
        risk_class=risk_class,
        overflow_detected=overflow_detected,
        blockage_pct=blockage_pct,
        elev=elev,
    )

    # Visual badge styling
    badge_colors = {
        "LOW": ("#22c55e", "#dcfce7"),
        "MODERATE": ("#f59e0b", "#fef3c7"),
        "HIGH": ("#f97316", "#ffedd5"),
        "CRITICAL": ("#ef4444", "#fee2e2"),
    }
    fg_col, bg_col = badge_colors.get(risk_class, ("#64748b", "#f1f5f9"))

    st.markdown(
        f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div>
                    <h3 style="margin: 0; color: #0f172a;">🛣️ {road_name}</h3>
                    <span style="color: #64748b; font-size: 0.85rem;">Segment ID: <code>{road_id}</code> | Type: {highway_class}</span>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.6rem; font-weight: bold; color: {fg_col};">{risk_score:.1f}<span style="font-size: 0.9rem; color: #94a3b8;">/100</span></div>
                    <span style="background: {bg_col}; color: {fg_col}; padding: 3px 10px; border-radius: 9999px; font-weight: 700; font-size: 0.8rem;">{risk_class}</span>
                </div>
            </div>
            <hr style="margin: 12px 0; border: none; border-top: 1px solid #f1f5f9;"/>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Structured 3-column metric cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🌧️ Rainfall & Hydrology**")
        st.write(f"• **24h Accumulation:** `{r24:.1f} mm`")
        st.write(f"• **1h Peak Rate:** `{r1:.1f} mm/h`")
        dist_w = float(_g("dist_to_water_m", 500.0))
        st.write(f"• **Dist to Waterbody:** `{dist_w:.0f} m`")

    with col2:
        st.markdown("**⛰️ Terrain & Topography**")
        st.write(f"• **Mean Elevation:** `{elev:.1f} m`")
        st.write(f"• **Slope:** `{slope:.1f}°`")
        st.write(f"• **Depression Context:** `{loc_ctx:+.1f} m`")

    with col3:
        st.markdown("**📹 Drainage & CCTV Status**")
        st.write(f"• **Drain Condition:** {drain_condition}")
        st.write(f"• **Inlet Blockage:** `{blockage_pct:.1f}%`")
        st.write(f"• **Water Coverage:** `{water_cov_pct:.1f}%`")
        if cctv_avail and cctv_conf is not None:
            st.caption(f"CCTV AI Confidence: {float(cctv_conf)*100.0:.0f}% | Delta: {delta:+.1f} pts")

    # Historical Inundation
    st.markdown("---")
    h_col1, h_col2 = st.columns([1, 2])
    with h_col1:
        st.markdown("**📜 Historical Vulnerability**")
        st.write(f"• **Prior Flood Events:** `{hist_count}`")
        st.write(f"• **Flood Frequency:** `{hist_freq:.2f}`")
        st.write(f"• **Evidence Tier:** `{hist_evidence}`")

    with h_col2:
        st.markdown("**💡 Operational Action / Recommendation**")
        st.info(recommendation)

    # Structured 'WHY IS THIS ROAD HIGH RISK?' explanation if high-risk or explanation dict available
    explanation_dict = road_data.get("explanation")
    if isinstance(explanation_dict, dict) and explanation_dict.get("is_high_risk"):
        st.markdown(
            f"""
            <div style="background: #fff7ed; border-left: 4px solid #f97316; padding: 12px 16px; border-radius: 6px; margin-top: 14px;">
                <h4 style="margin: 0 0 6px 0; color: #c2410c;">❓ {explanation_dict.get('headline', 'WHY IS THIS ROAD HIGH RISK?')}</h4>
                <p style="margin: 0 0 8px 0; font-size: 0.9rem; color: #431407;">{explanation_dict.get('summary_text', '')}</p>
                <div style="display: flex; flex-wrap: wrap; gap: 6px;">
            """,
            unsafe_allow_html=True,
        )
        for f in explanation_dict.get("factors", []):
            st.markdown(
                f"<span style='background: #ffedd5; color: #9a3412; font-weight: 600; font-size: 0.8rem; padding: 2px 8px; border-radius: 4px; border: 1px solid #fed7aa;'>• {f}</span>",
                unsafe_allow_html=True,
            )
        st.markdown("</div></div>", unsafe_allow_html=True)
