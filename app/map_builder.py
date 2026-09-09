"""
Interactive Folium Map Builder for Chennai Flood Access Risk Mapper.
Constructs rich multi-layer maps supporting:
- Road flood-risk visualization (LOW, MODERATE, HIGH, CRITICAL)
- Stormwater drainage network
- River & canal network
- Water bodies & wetlands
- DrainWatch CCTV camera stations
- Normal vs. Flood-safe route comparison polylines
"""
from typing import Dict, Any, List, Optional, Tuple
import folium
from folium import plugins
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Polygon, Point

from config.constants import (
    CHENNAI_CENTER,
    RISK_COLORS,
    RISK_CLASSES,
)


def get_risk_color(score: float) -> str:
    """Map a 0-100 risk score to its standard risk class color."""
    if score >= 75.0:
        return "#ef4444"  # CRITICAL - Red
    elif score >= 50.0:
        return "#f97316"  # HIGH - Orange
    elif score >= 25.0:
        return "#f59e0b"  # MODERATE - Amber
    else:
        return "#22c55e"  # LOW - Green


def create_chennai_base_map(
    center: Tuple[float, float] = CHENNAI_CENTER,
    zoom_start: int = 12,
) -> folium.Map:
    """Create a high-contrast base Folium map."""
    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True,
    )
    return m


def add_water_layers(
    folium_map: folium.Map,
    waterways_gdf: gpd.GeoDataFrame,
    waterbodies_gdf: gpd.GeoDataFrame,
    show: bool = True,
) -> None:
    """Add rivers, canals, lakes, and wetlands to the map."""
    # 1. Water bodies (Polygons)
    fg_waterbodies = folium.FeatureGroup(name="🌊 Water Bodies & Wetlands", show=show)
    for _, row in waterbodies_gdf.iterrows():
        geom = row.geometry
        if isinstance(geom, Polygon):
            coords = [(lat, lon) for lon, lat in geom.exterior.coords]
            w_type = row.get("type", "LAKE")
            w_name = row.get("name", "Water Body")
            color = "#0284c7" if w_type == "RESERVOIR" else "#0ea5e9" if w_type == "LAKE" else "#059669"
            fill_color = color

            popup_html = f"""
            <div style="font-family: sans-serif; min-width: 140px;">
                <b style="color: #0369a1;">{w_name}</b><br/>
                <span style="font-size: 0.85rem; color: #64748b;">Type: {w_type}</span>
            </div>
            """
            folium.Polygon(
                locations=coords,
                color=color,
                weight=2,
                fill=True,
                fill_color=fill_color,
                fill_opacity=0.35,
                tooltip=f"{w_name} ({w_type})",
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(fg_waterbodies)
    fg_waterbodies.add_to(folium_map)

    # 2. Rivers & Canals (Lines)
    fg_waterways = folium.FeatureGroup(name="💧 Rivers & Canals", show=show)
    for _, row in waterways_gdf.iterrows():
        geom = row.geometry
        if isinstance(geom, LineString):
            coords = [(lat, lon) for lon, lat in geom.coords]
            w_name = row.get("name", "Waterway")
            w_type = row.get("type", "RIVER")
            color = "#1e40af" if w_type == "RIVER" else "#2563eb"
            weight = 4 if w_type == "RIVER" else 3

            popup_html = f"""
            <div style="font-family: sans-serif; min-width: 160px;">
                <b style="color: #1e3a8a;">{w_name}</b><br/>
                <span style="font-size: 0.85rem; color: #475569;">Classification: {w_type}</span>
            </div>
            """
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=weight,
                opacity=0.85,
                tooltip=f"{w_name} ({w_type})",
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(fg_waterways)
    fg_waterways.add_to(folium_map)


def add_drainage_layer(
    folium_map: folium.Map,
    drains_gdf: gpd.GeoDataFrame,
    show: bool = True,
) -> None:
    """Add stormwater drainage network lines."""
    fg_drains = folium.FeatureGroup(name="🌐 Stormwater Drainage Lines", show=show)
    for _, row in drains_gdf.iterrows():
        geom = row.geometry
        if isinstance(geom, LineString):
            coords = [(lat, lon) for lon, lat in geom.coords]
            d_name = row.get("name", "Storm Drain")
            d_id = row.get("drain_id", "")
            outfall = row.get("outfall", "Channel")
            cap = row.get("capacity_m3_s", 15.0)

            popup_html = f"""
            <div style="font-family: sans-serif; min-width: 180px;">
                <b style="color: #0e7490;">{d_name}</b><br/>
                <span style="font-size: 0.8rem; color: #64748b;">ID: {d_id}</span><br/>
                <span style="font-size: 0.85rem;">Discharge: <b>{cap} m³/s</b></span><br/>
                <span style="font-size: 0.85rem;">Outfall: {outfall}</span>
            </div>
            """
            folium.PolyLine(
                locations=coords,
                color="#06b6d4",
                weight=3.5,
                dash_array="6, 4",
                opacity=0.9,
                tooltip=f"Drain: {d_name} ({cap} m³/s)",
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(fg_drains)
    fg_drains.add_to(folium_map)


def add_road_risk_layer(
    folium_map: folium.Map,
    scored_roads_gdf: gpd.GeoDataFrame,
    selected_road_id: Optional[str] = None,
    show: bool = True,
) -> None:
    """Add road polylines colored by final flood-access risk score (0-100)."""
    fg_roads = folium.FeatureGroup(name="🛣️ Road Flood-Risk Layer", show=show)

    for _, row in scored_roads_gdf.iterrows():
        geom = row.geometry
        if isinstance(geom, LineString):
            coords = [(lat, lon) for lon, lat in geom.coords]
            seg_id = str(row.get("segment_id", row.get("road_id", "")))
            name = str(row.get("name", "Chennai Road"))
            score = float(row.get("flood_access_risk_score", row.get("base_risk_score", 0.0)))
            r_class = str(row.get("risk_class", "LOW")).upper()
            delta = float(row.get("drainwatch_delta", 0.0))
            elev = float(row.get("elevation_mean", 10.0))
            loc_ctx = float(row.get("local_elevation_context", 0.0))
            cctv_avail = bool(row.get("drainwatch_available", False))

            color = get_risk_color(score)
            is_selected = (seg_id == selected_road_id)
            weight = 7 if is_selected else 5
            opacity = 1.0 if is_selected else 0.85

            popup_html = f"""
            <div style="font-family: sans-serif; min-width: 220px; font-size: 0.9rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <b style="color: #0f172a;">{name}</b>
                </div>
                <div style="color: #64748b; font-size: 0.75rem; margin-bottom: 6px;">ID: {seg_id}</div>
                <div style="margin-bottom: 4px;">
                    Risk Score: <b style="color: {color}; font-size: 1.1rem;">{score:.1f}/100</b>
                    <span style="background: {color}22; color: {color}; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.75rem; margin-left: 6px;">{r_class}</span>
                </div>
                <hr style="margin: 4px 0; border: none; border-top: 1px solid #e2e8f0;"/>
                <div style="font-size: 0.8rem; color: #334155; line-height: 1.4;">
                    • Mean Elevation: <b>{elev:.1f} m</b> ({loc_ctx:+.1f}m context)<br/>
                    • CCTV Adjustment: <b>{delta:+.1f} pts</b> ({'Active' if cctv_avail else 'No Camera'})<br/>
                    • Road Class: {str(row.get('highway', 'primary')).capitalize()}
                </div>
            </div>
            """

            folium.PolyLine(
                locations=coords,
                color=color,
                weight=weight,
                opacity=opacity,
                tooltip=f"{name} — Risk: {score:.1f}/100 ({r_class})",
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(fg_roads)

    fg_roads.add_to(folium_map)


def add_cctv_camera_markers(
    folium_map: folium.Map,
    cctv_list: List[Dict[str, Any]],
    show: bool = True,
) -> None:
    """Add verified DrainWatch AI CCTV camera location markers."""
    fg_cctv = folium.FeatureGroup(name="📹 DrainWatch CCTV Cameras", show=show)

    for cam in cctv_list:
        lat = cam["lat"]
        lon = cam["lon"]
        cam_id = cam["camera_id"]
        cam_name = cam["name"]
        blk_pct = cam["blockage_percentage"]
        w_pct = cam["water_coverage_percentage"]
        overflow = cam["overflow_detected"]
        conf = cam["confidence"]
        status = cam.get("status", "NORMAL")
        cond_class = cam.get("condition_class", "drain_clear").replace("_", " ").title()

        # Marker color based on severity
        if overflow or status == "CRITICAL_ALERT" or blk_pct >= 70.0:
            marker_color = "red"
            icon_name = "exclamation-triangle"
        elif blk_pct >= 40.0 or status == "WARNING":
            marker_color = "orange"
            icon_name = "video-camera"
        else:
            marker_color = "green"
            icon_name = "video-camera"

        popup_html = f"""
        <div style="font-family: sans-serif; min-width: 220px; font-size: 0.9rem;">
            <div style="font-weight: bold; color: #0f172a; font-size: 1rem;">📹 {cam_name}</div>
            <div style="color: #64748b; font-size: 0.75rem; margin-bottom: 6px;">ID: {cam_id}</div>
            <hr style="margin: 4px 0; border: none; border-top: 1px solid #e2e8f0;"/>
            <div style="font-size: 0.85rem; line-height: 1.45; color: #1e293b;">
                • <b>Condition:</b> {cond_class}<br/>
                • <b>Inlet Blockage:</b> <b style="color: {'#dc2626' if blk_pct>=50 else '#16a34a'};">{blk_pct:.1f}%</b><br/>
                • <b>Water Coverage:</b> <b>{w_pct:.1f}%</b><br/>
                • <b>Drain Overflow:</b> <b style="color: {'#dc2626' if overflow else '#16a34a'};">{'YES (CRITICAL)' if overflow else 'NO'}</b><br/>
                • <b>AI Confidence:</b> <b>{conf * 100.0:.1f}%</b>
            </div>
        </div>
        """

        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color=marker_color, icon=icon_name, prefix="fa"),
            tooltip=f"CCTV: {cam_id} ({cond_class} - {blk_pct:.0f}% Blockage)",
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(fg_cctv)

    fg_cctv.add_to(folium_map)


def add_route_comparison_overlay(
    folium_map: folium.Map,
    normal_coords: List[Tuple[float, float]],
    flood_safe_coords: List[Tuple[float, float]],
    origin_coord: Tuple[float, float],
    dest_coord: Tuple[float, float],
    origin_name: str = "Origin",
    dest_name: str = "Destination",
    normal_metrics: Optional[Dict[str, Any]] = None,
    safe_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Overlay Normal vs. Flood-Safe Navigation Routes."""
    fg_routes = folium.FeatureGroup(name="🧭 Route Comparison Overlay", show=True)

    # 1. Normal Shortest Route (Blue Dashed)
    if normal_coords and len(normal_coords) >= 2:
        norm_dist = normal_metrics.get("distance_km", 0.0) if normal_metrics else 0.0
        norm_time = normal_metrics.get("estimated_travel_time_minutes", 0.0) if normal_metrics else 0.0
        norm_risk = normal_metrics.get("mean_risk_score", 0.0) if normal_metrics else 0.0

        folium.PolyLine(
            locations=normal_coords,
            color="#3b82f6",
            weight=6,
            opacity=0.8,
            dash_array="8, 6",
            tooltip=f"NORMAL ROUTE: {norm_dist:.1f} km | {norm_time:.1f} min | Risk: {norm_risk:.1f}/100",
            popup=folium.Popup(f"<b>NORMAL SHORTEST ROUTE</b><br/>Distance: {norm_dist:.1f} km<br/>Time: {norm_time:.1f} min<br/>Mean Risk: {norm_risk:.1f}/100", max_width=250),
        ).add_to(fg_routes)

    # 2. Flood-Safe Alternative Route (Emerald Solid)
    if flood_safe_coords and len(flood_safe_coords) >= 2:
        safe_dist = safe_metrics.get("distance_km", 0.0) if safe_metrics else 0.0
        safe_time = safe_metrics.get("estimated_travel_time_minutes", 0.0) if safe_metrics else 0.0
        safe_risk = safe_metrics.get("mean_risk_score", 0.0) if safe_metrics else 0.0

        folium.PolyLine(
            locations=flood_safe_coords,
            color="#10b981",
            weight=7,
            opacity=0.95,
            tooltip=f"FLOOD-SAFE ROUTE: {safe_dist:.1f} km | {safe_time:.1f} min | Risk: {safe_risk:.1f}/100",
            popup=folium.Popup(f"<b>FLOOD-SAFE ROUTE</b><br/>Distance: {safe_dist:.1f} km<br/>Time: {safe_time:.1f} min<br/>Mean Risk: {safe_risk:.1f}/100", max_width=250),
        ).add_to(fg_routes)

    # 3. Origin & Destination Markers
    folium.Marker(
        location=origin_coord,
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
        tooltip=f"Origin: {origin_name}",
        popup=f"<b>START: {origin_name}</b>",
    ).add_to(fg_routes)

    folium.Marker(
        location=dest_coord,
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
        tooltip=f"Destination: {dest_name}",
        popup=f"<b>DESTINATION: {dest_name}</b>",
    ).add_to(fg_routes)

    fg_routes.add_to(folium_map)


def build_full_dashboard_map(
    scored_roads_gdf: gpd.GeoDataFrame,
    waterways_gdf: gpd.GeoDataFrame,
    waterbodies_gdf: gpd.GeoDataFrame,
    drains_gdf: gpd.GeoDataFrame,
    cctv_list: List[Dict[str, Any]],
    selected_road_id: Optional[str] = None,
    route_comparison_data: Optional[Dict[str, Any]] = None,
) -> folium.Map:
    """Build unified interactive multi-layer Chennai map."""
    m = create_chennai_base_map()

    # Base layers
    add_water_layers(m, waterways_gdf=waterways_gdf, waterbodies_gdf=waterbodies_gdf, show=True)
    add_drainage_layer(m, drains_gdf=drains_gdf, show=True)
    add_road_risk_layer(m, scored_roads_gdf=scored_roads_gdf, selected_road_id=selected_road_id, show=True)
    add_cctv_camera_markers(m, cctv_list=cctv_list, show=True)

    # Optional Route Comparison Overlay
    if route_comparison_data and route_comparison_data.get("has_route"):
        add_route_comparison_overlay(
            folium_map=m,
            normal_coords=route_comparison_data.get("normal_coords", []),
            flood_safe_coords=route_comparison_data.get("safe_coords", []),
            origin_coord=route_comparison_data.get("origin_coord", CHENNAI_CENTER),
            dest_coord=route_comparison_data.get("dest_coord", CHENNAI_CENTER),
            origin_name=route_comparison_data.get("origin_name", "Origin"),
            dest_name=route_comparison_data.get("dest_name", "Destination"),
            normal_metrics=route_comparison_data.get("normal_metrics"),
            safe_metrics=route_comparison_data.get("safe_metrics"),
        )

    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    return m
