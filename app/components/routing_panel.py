"""
Flood-Aware Road Routing Panel Component for Streamlit Dashboard.
Compares Normal Shortest Route vs. Flood-Safe Detour Route across:
- Travel distance (km)
- Estimated travel time (minutes)
- Risk score (mean & max)
- High-risk road segments avoided
- Additional distance & time delta
- Vehicle-specific clearance depth modulation
"""
from typing import Dict, Any, List, Optional, Tuple
import streamlit as st
import pandas as pd
import geopandas as gpd

from routing.graph import RoadGraphManager
from routing.router import FloodAwareRouter, RouteComparison, RouteDetails
from config.constants import DEPTH_THRESHOLDS_CM


LOCATION_PRESETS: Dict[str, Tuple[str, str]] = {
    "Chennai Central ➔ Chennai Airport": ("node_central", "node_airport"),
    "Koyambedu CMBT ➔ Marina Beach": ("node_koyambedu", "node_marina"),
    "Anna Nagar ➔ Velachery Junction": ("node_annanagar", "node_velachery"),
    "T. Nagar ➔ OMR Tidel Park": ("node_tnagar", "node_omr_tidel"),
    "Perambur ➔ Guindy Kathipara": ("node_perambur", "node_guindy"),
    "Porur Junction ➔ Chennai Central": ("node_porur", "node_central"),
    "Velachery Junction ➔ Chennai Airport": ("node_velachery", "node_airport"),
}

NODE_NAMES: Dict[str, str] = {
    "node_central": "Chennai Central Railway Station",
    "node_parrys": "Parrys Corner / Broadway",
    "node_marina": "Marina Beach / Kamarajar Salai",
    "node_tnagar": "T. Nagar Panagal Park",
    "node_guindy": "Guindy Kathipara Junction",
    "node_airport": "Chennai International Airport",
    "node_velachery": "Velachery 100ft Junction",
    "node_omr_tidel": "OMR Tidel Park Tharamani",
    "node_adyar": "Adyar Bridge Maraimalai Adigal",
    "node_koyambedu": "Koyambedu CMBT Terminal",
    "node_annanagar": "Anna Nagar Roundtana",
    "node_kilpauk": "Kilpauk EVR Periyar Salai",
    "node_perambur": "Perambur High Road Underpass",
    "node_ecr_thiru": "ECR Thiruvanmiyur Junction",
    "node_porur": "Porur Junction",
    "node_tambaram": "Tambaram Sanatorium",
}


def render_routing_panel(
    scored_roads_gdf: gpd.GeoDataFrame,
) -> Optional[Dict[str, Any]]:
    """
    Render flood-aware route planning controls and side-by-side comparison.
    Returns route coordinates and metrics for map overlay.
    """
    st.markdown("### 🧭 Flood-Aware Route Planner")
    st.caption("Compare conventional shortest paths against flood-safe alternative detours in real time.")

    col1, col2, col3 = st.columns([1.5, 1.2, 1.0])

    with col1:
        route_preset = st.selectbox(
            "Quick Route Preset",
            list(LOCATION_PRESETS.keys()),
            index=0,
        )
        origin_node, dest_node = LOCATION_PRESETS[route_preset]

    with col2:
        vehicle_type = st.selectbox(
            "Vehicle Type (Clearance)",
            ["car", "motorcycle", "suv", "bus", "pedestrian"],
            index=0,
            format_func=lambda x: f"{x.capitalize()} (Max Depth: {DEPTH_THRESHOLDS_CM.get(x, 25)} cm)",
        )

    with col3:
        algo = st.selectbox("Routing Algorithm", ["Dijkstra (Exact)", "A* (Heuristic)"], index=0)
        algo_name = "dijkstra" if "Dijkstra" in algo else "astar"

    # Build Graph Manager & Router
    graph_manager = RoadGraphManager()
    graph_manager.build_from_geodataframe(scored_roads_gdf, risk_column="flood_access_risk_score")
    router = FloodAwareRouter(graph_manager, algorithm=algo_name, high_risk_threshold=50.0)

    try:
        comparison: RouteComparison = router.compare_routes(origin_node, dest_node)
    except Exception as e:
        st.warning(f"Unable to compute route: {str(e)}")
        return None

    # Side-by-side Comparison Cards
    st.markdown("---")
    c_norm, c_safe, c_delta = st.columns(3)

    norm = comparison.normal_route
    safe = comparison.flood_safe_route

    with c_norm:
        st.markdown(
            f"""
            <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px;">
                <div style="font-weight: 700; color: #1e40af; font-size: 1rem; margin-bottom: 4px;">🔵 NORMAL SHORTEST ROUTE</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 8px;">Standard Travel-Time Optimized</div>
                <div style="font-size: 1.3rem; font-weight: bold; color: #0f172a;">{norm.distance_km:.1f} km <span style="font-size: 0.9rem; font-weight: normal; color: #64748b;">({norm.estimated_travel_time_minutes:.1f} min)</span></div>
                <hr style="margin: 8px 0; border: none; border-top: 1px solid #e2e8f0;"/>
                <div style="font-size: 0.85rem; line-height: 1.4;">
                    • <b>Mean Risk Score:</b> <span style="color: {'#dc2626' if norm.mean_risk_score>=50 else '#16a34a'}; font-weight: bold;">{norm.mean_risk_score:.1f}/100</span><br/>
                    • <b>Max Segment Risk:</b> <b>{norm.max_risk_score:.1f}/100</b><br/>
                    • <b>High-Risk Segments:</b> <b style="color: {'#dc2626' if norm.number_of_high_risk_segments>0 else '#16a34a'};">{norm.number_of_high_risk_segments}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_safe:
        st.markdown(
            f"""
            <div style="background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 8px; padding: 12px;">
                <div style="font-weight: 700; color: #065f46; font-size: 1rem; margin-bottom: 4px;">🟢 FLOOD-SAFE ALTERNATIVE</div>
                <div style="font-size: 0.8rem; color: #059669; margin-bottom: 8px;">Inundation-Penalized Detour</div>
                <div style="font-size: 1.3rem; font-weight: bold; color: #064e3b;">{safe.distance_km:.1f} km <span style="font-size: 0.9rem; font-weight: normal; color: #059669;">({safe.estimated_travel_time_minutes:.1f} min)</span></div>
                <hr style="margin: 8px 0; border: none; border-top: 1px solid #d1fae5;"/>
                <div style="font-size: 0.85rem; line-height: 1.4; color: #064e3b;">
                    • <b>Mean Risk Score:</b> <span style="color: #059669; font-weight: bold;">{safe.mean_risk_score:.1f}/100</span><br/>
                    • <b>Max Segment Risk:</b> <b>{safe.max_risk_score:.1f}/100</b><br/>
                    • <b>High-Risk Segments:</b> <b>{safe.number_of_high_risk_segments}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_delta:
        risk_drop = comparison.risk_score_reduction
        avoided = comparison.high_risk_segments_avoided
        extra_km = comparison.additional_distance_km
        extra_min = comparison.additional_travel_time_minutes

        reroute_badge = (
            "<span style='background: #fee2e2; color: #dc2626; padding: 4px 8px; border-radius: 6px; font-weight: bold;'>⚠️ REROUTE RECOMMENDED</span>"
            if comparison.reroute_recommended
            else "<span style='background: #dcfce7; color: #166534; padding: 4px 8px; border-radius: 6px; font-weight: bold;'>✅ NORMAL PATH SAFE</span>"
        )

        st.markdown(
            f"""
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px;">
                <div style="margin-bottom: 8px;">{reroute_badge}</div>
                <div style="font-size: 0.85rem; line-height: 1.45; color: #334155;">
                    • <b>Risk Reduction:</b> <b style="color: #16a34a;">{risk_drop:.1f}%</b><br/>
                    • <b>High-Risk Avoided:</b> <b>{avoided} segment(s)</b><br/>
                    • <b>Additional Distance:</b> +{extra_km:.2f} km<br/>
                    • <b>Additional Travel Time:</b> +{extra_min:.1f} min
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Narrative Summary
    st.info(f"📋 **Route Comparison Summary:** {comparison.summary}")

    # Extract coordinates for map display
    def _extract_coords(node_list: List[str]) -> List[Tuple[float, float]]:
        coords = []
        for n in node_list:
            c = graph_manager.node_coords.get(n)
            if c:
                coords.append(c)
        return coords

    norm_coords = _extract_coords(norm.nodes)
    safe_coords = _extract_coords(safe.nodes)
    origin_pt = graph_manager.node_coords.get(origin_node, (13.0827, 80.2707))
    dest_pt = graph_manager.node_coords.get(dest_node, (12.9820, 80.1650))

    return {
        "has_route": True,
        "normal_coords": norm_coords,
        "safe_coords": safe_coords,
        "origin_coord": origin_pt,
        "dest_coord": dest_pt,
        "origin_name": NODE_NAMES.get(origin_node, origin_node),
        "dest_name": NODE_NAMES.get(dest_node, dest_node),
        "normal_metrics": norm.__dict__,
        "safe_metrics": safe.__dict__,
        "comparison": comparison.__dict__,
    }
