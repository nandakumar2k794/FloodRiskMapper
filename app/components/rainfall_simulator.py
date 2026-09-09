"""
Rainfall Scenario Simulator Component for Streamlit Dashboard.
Allows interactive simulation of meteorological monsoon scenarios:
- Normal Monsoon
- Moderate Rainfall
- Heavy Monsoon (IMD Yellow/Orange Alert)
- Very Heavy Rainfall (Red Alert)
- Extreme Cloudburst / Michaung Scale
- Custom 24h and 1h rainfall accumulation sliders
"""
from typing import Dict, Any, Tuple
import streamlit as st
import pandas as pd

from config.constants import RAINFALL_THRESHOLDS_MM


RAINFALL_PRESETS: Dict[str, Dict[str, float]] = {
    "🌤️ Light / Baseline (10 mm / 24h)": {
        "rainfall_15min": 2.0,
        "rainfall_1h": 4.0,
        "rainfall_3h": 6.0,
        "rainfall_6h": 8.0,
        "rainfall_24h": 10.0,
        "intensity_label": "Light Rain",
    },
    "🌧️ Moderate Monsoon (40 mm / 24h)": {
        "rainfall_15min": 5.0,
        "rainfall_1h": 12.0,
        "rainfall_3h": 22.0,
        "rainfall_6h": 30.0,
        "rainfall_24h": 40.0,
        "intensity_label": "Moderate Rainfall",
    },
    "⚠️ Heavy Rainfall - IMD Heavy Alert (85 mm / 24h)": {
        "rainfall_15min": 10.0,
        "rainfall_1h": 25.0,
        "rainfall_3h": 48.0,
        "rainfall_6h": 65.0,
        "rainfall_24h": 85.0,
        "intensity_label": "Heavy Rainfall (IMD)",
    },
    "🚨 Very Heavy Rainfall - Red Alert (160 mm / 24h)": {
        "rainfall_15min": 18.0,
        "rainfall_1h": 45.0,
        "rainfall_3h": 90.0,
        "rainfall_6h": 125.0,
        "rainfall_24h": 160.0,
        "intensity_label": "Very Heavy (Inundation Likely)",
    },
    "🌊 Cyclone Michaung / Extreme Inundation (280 mm / 24h)": {
        "rainfall_15min": 25.0,
        "rainfall_1h": 65.0,
        "rainfall_3h": 130.0,
        "rainfall_6h": 200.0,
        "rainfall_24h": 280.0,
        "intensity_label": "Extremely Heavy (Widespread Flooding)",
    },
}


def render_rainfall_simulator() -> Dict[str, Any]:
    """
    Render rainfall simulator controls in the Streamlit UI.
    Returns rainfall dictionary matching ML Stage 1 schema.
    """
    st.markdown("### 🌧️ Rainfall Scenario Simulator")
    st.caption("Broadcast live simulated rainfall accumulation across the Chennai road network.")

    col1, col2 = st.columns([1.5, 1])

    with col1:
        preset_choice = st.selectbox(
            "Select Scenario Preset",
            list(RAINFALL_PRESETS.keys()),
            index=2,  # Default to Heavy Monsoon (85mm)
        )
        preset_values = RAINFALL_PRESETS[preset_choice]

    with col2:
        source_mode = st.radio(
            "Rainfall Feed Mode",
            ["Simulated Scenario", "Dynamic Slider Override"],
            horizontal=True,
        )

    if source_mode == "Dynamic Slider Override":
        c_r1, c_r2 = st.columns(2)
        with c_r1:
            r24 = st.slider("24-Hour Cumulative Rainfall (mm)", 0.0, 400.0, float(preset_values["rainfall_24h"]), 5.0)
        with c_r2:
            r1 = st.slider("1-Hour Peak Intensity (mm/h)", 0.0, 100.0, min(80.0, float(preset_values["rainfall_1h"])), 2.0)

        r15 = round(r1 * 0.35, 1)
        r3 = round(min(r24, r1 * 2.2), 1)
        r6 = round(min(r24, r1 * 3.5), 1)
        source_type = "SIMULATED_SLIDER"
    else:
        r24 = float(preset_values["rainfall_24h"])
        r1 = float(preset_values["rainfall_1h"])
        r15 = float(preset_values["rainfall_15min"])
        r3 = float(preset_values["rainfall_3h"])
        r6 = float(preset_values["rainfall_6h"])
        source_type = "SIMULATED_PRESET"

    # IMD Warning Badge
    if r24 >= 204.5:
        badge_color = "#ef4444"
        badge_text = "EXTREMELY HEAVY (RED ALERT)"
    elif r24 >= 115.6:
        badge_color = "#ea580c"
        badge_text = "VERY HEAVY (ORANGE ALERT)"
    elif r24 >= 64.5:
        badge_color = "#f59e0b"
        badge_text = "HEAVY RAINFALL (YELLOW ALERT)"
    else:
        badge_color = "#10b981"
        badge_text = "NORMAL / MODERATE"

    st.markdown(
        f"""
        <div style="background: {badge_color}18; border-left: 4px solid {badge_color}; padding: 8px 14px; border-radius: 6px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <b style="color: {badge_color};">{badge_text}</b> — 24h Rain: <b>{r24:.1f} mm</b> | 1h Peak: <b>{r1:.1f} mm/h</b>
            </div>
            <div style="font-size: 0.8rem; color: #64748b;">Source: {source_type}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return {
        "rainfall_15min": r15,
        "rainfall_1h": r1,
        "rainfall_3h": r3,
        "rainfall_6h": r6,
        "rainfall_24h": r24,
        "rainfall_source_type": source_type,
    }


def get_rainfall_dataframe(rainfall_dict: Dict[str, Any]) -> pd.DataFrame:
    """Format single broadcast row as DataFrame for ML feature matrix."""
    return pd.DataFrame([rainfall_dict])
