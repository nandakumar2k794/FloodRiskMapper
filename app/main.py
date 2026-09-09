"""
Chennai Flood Access Risk Mapper with DrainWatch AI
Interactive Streamlit Operational Dashboard

Implements all 10 Core Dashboard Requirements:
1. Chennai interactive map
2. Road flood-risk visualization (LOW, MODERATE, HIGH, CRITICAL)
3. Drainage layers
4. River / canal layers
5. Water bodies & wetlands
6. CCTV locations
7. Selected-road details (Road ID, Risk, Rain, Elev, Slope, Drain Condition, Blockage, Water Coverage, Historical Risk, Recommendation)
8. DrainWatch panel (Camera, Garbage, Blockage, Water, Overflow, Confidence)
9. Rainfall scenario simulator
10. Normal vs. Flood-safe route comparison
"""
import sys
from pathlib import Path

# Add project root directory to sys.path to enable imports from config, app, etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, List, Optional
import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import geopandas as gpd

from config.constants import (
    PROJECT_NAME,
    PROJECT_VERSION,
    CHENNAI_CENTER,
    RISK_CLASSES,
    RISK_COLORS,
)
from app.data_loader import (
    get_chennai_roads_network,
    get_chennai_waterways,
    get_chennai_waterbodies,
    get_chennai_drains,
    get_chennai_cctv_cameras,
    get_cctv_dataframe,
)
from app.map_builder import build_full_dashboard_map
from app.components.rainfall_simulator import render_rainfall_simulator, get_rainfall_dataframe
from app.components.road_details import render_selected_road_panel
from app.components.drainwatch_panel import render_drainwatch_panel
from app.components.routing_panel import render_routing_panel
from ml.risk_scorer import RoadFloodRiskScorer


# Page Setup
st.set_page_config(
    page_title="Chennai Flood Access Risk Mapper | DrainWatch AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .main-title {
            font-size: 1.8rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0px;
        }
        .sub-title {
            font-size: 0.95rem;
            color: #64748b;
            margin-bottom: 12px;
        }
        .kpi-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 10px 14px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_base_gis_layers():
    """Load and cache static GIS layers."""
    roads = get_chennai_roads_network()
    waterways = get_chennai_waterways()
    waterbodies = get_chennai_waterbodies()
    drains = get_chennai_drains()
    cctv = get_chennai_cctv_cameras()
    return roads, waterways, waterbodies, drains, cctv


def score_roads_live(
    roads_gdf: gpd.GeoDataFrame,
    rainfall_dict: Dict[str, Any],
    cctv_df: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Run live end-to-end ML scoring pipeline on road network."""
    scorer = RoadFloodRiskScorer()
    rain_df = get_rainfall_dataframe(rainfall_dict)
    scored_gdf = scorer.score_network(
        roads_gdf=roads_gdf,
        rainfall_df=rain_df,
        drainwatch_df=cctv_df,
    )
    return scored_gdf


def main():
    # 1. Header Banner
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 12px;">
            <div>
                <div class="main-title">🌊 {PROJECT_NAME}</div>
                <div class="sub-title">Real-Time Street-Level Flood Accessibility & DrainWatch Vision Intelligence</div>
            </div>
            <div style="text-align: right;">
                <span style="background: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.8rem;">v{PROJECT_VERSION}</span>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">Zone 10-13 | GCC / CMWSSB Basin</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Load GIS Data
    base_roads, waterways, waterbodies, drains, cctv_list = load_base_gis_layers()
    cctv_df = get_cctv_dataframe()

    # 3. Sidebar Controls
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/flood.png", width=56)
        st.markdown("### 🎛️ Scenario Controls")
        
        # Rainfall Simulator in Sidebar / Top
        rainfall_dict = render_rainfall_simulator()
        st.markdown("---")

        st.markdown("### 🔍 Road Inspector")
        road_names_map = {
            r["segment_id"]: f"{r['name']} ({r['segment_id']})"
            for _, r in base_roads.iterrows()
        }
        selected_road_id = st.selectbox(
            "Select Road to Inspect",
            list(road_names_map.keys()),
            index=0,
            format_func=lambda x: road_names_map[x],
        )

        st.markdown("---")
        st.caption(
            "**Defensibility Assurance:**\n"
            "• Physical monotonicity enforced.\n"
            "• CCTV is additive delta only.\n"
            "• Zero feature double-counting."
        )

    # 4. Live ML Network Scoring
    scored_roads = score_roads_live(base_roads, rainfall_dict, cctv_df)

    # 5. Top KPI Summary Bar
    k1, k2, k3, k4, k5 = st.columns(5)
    
    total_roads = len(scored_roads)
    high_critical_count = len(scored_roads[scored_roads["flood_access_risk_score"] >= 50.0])
    critical_count = len(scored_roads[scored_roads["flood_access_risk_score"] >= 75.0])
    mean_risk = float(scored_roads["flood_access_risk_score"].mean())
    active_cameras = len(cctv_list)

    with k1:
        st.metric("Monitored Corridors", f"{total_roads} Segments", delta="100% Coverage")
    with k2:
        st.metric("High/Critical Risk", f"{high_critical_count} Roads", delta=f"{critical_count} Inaccessible", delta_color="inverse")
    with k3:
        st.metric("Mean Network Risk", f"{mean_risk:.1f} / 100", delta=f"Rain: {rainfall_dict['rainfall_24h']}mm")
    with k4:
        st.metric("DrainWatch CCTV", f"{active_cameras} Stations", delta="Active Feeds")
    with k5:
        st.metric("Active IMD Alert", "Heavy Inundation" if rainfall_dict["rainfall_24h"] >= 64.5 else "Moderate", delta="Monsoon Watch")

    st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

    # 6. Main Navigation Tabs
    tab_map, tab_road, tab_drainwatch, tab_routing = st.tabs([
        "🗺️ Chennai Interactive Risk Map",
        "📋 Selected Road Details & Diagnostics",
        "👁️ DrainWatch AI Vision Telemetry",
        "🧭 Flood-Aware Route Planner",
    ])

    # -------------------------------------------------------------
    # TAB 1: Chennai Interactive Map
    # -------------------------------------------------------------
    with tab_map:
        st.markdown("##### 📍 Interactive Multi-Layer Map (Road Risk, Drainage, Waterways & CCTV)")
        
        # Optional route overlay data from session state
        route_overlay = st.session_state.get("active_route_data", None)

        folium_map = build_full_dashboard_map(
            scored_roads_gdf=scored_roads,
            waterways_gdf=waterways,
            waterbodies_gdf=waterbodies,
            drains_gdf=drains,
            cctv_list=cctv_list,
            selected_road_id=selected_road_id,
            route_comparison_data=route_overlay,
        )

        st_folium(folium_map, width=None, height=540, use_container_width=True)

        # Legend Bar
        l1, l2, l3, l4 = st.columns(4)
        with l1:
            st.markdown("🟢 **LOW (0–25)**: Normal passage")
        with l2:
            st.markdown("🟡 **MODERATE (25–50)**: Standard caution")
        with l3:
            st.markdown("🟠 **HIGH (50–75)**: Reroute 2-wheelers")
        with l4:
            st.markdown("🔴 **CRITICAL (75–100)**: Impassable")

    # -------------------------------------------------------------
    # TAB 2: Selected Road Details
    # -------------------------------------------------------------
    with tab_road:
        selected_row = scored_roads[scored_roads["segment_id"] == selected_road_id]
        if not selected_row.empty:
            road_data_dict = selected_row.iloc[0].to_dict()
            render_selected_road_panel(road_data_dict, rainfall_data=rainfall_dict)
        else:
            st.warning("Please select a valid road from the sidebar dropdown.")

    # -------------------------------------------------------------
    # TAB 3: DrainWatch AI Vision Panel
    # -------------------------------------------------------------
    with tab_drainwatch:
        render_drainwatch_panel(cctv_list=cctv_list)

    # -------------------------------------------------------------
    # TAB 4: Flood-Aware Route Planner
    # -------------------------------------------------------------
    with tab_routing:
        route_res = render_routing_panel(scored_roads_gdf=scored_roads)
        if route_res:
            st.session_state["active_route_data"] = route_res
            st.success("✅ Navigation computed! Switch to **🗺️ Chennai Interactive Risk Map** to view the dual route overlay.")


if __name__ == "__main__":
    main()
