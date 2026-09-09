"""
Risk Map API route — returns GeoJSON FeatureCollection of road segments
with current flood risk scores.
"""
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse

from src.api.app_state import get_predictor
from src.db.mongo import get_db
from config.constants import RISK_COLORS, RISK_CLASSES, CHENNAI_CENTER

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/risk-map", summary="Get road segment risk GeoJSON")
async def get_risk_map(
    rainfall: float = Query(120.0, ge=0.0, le=400.0, description="Rainfall accumulation in mm/24h"),
    risk_class: Optional[str] = Query(None, description="Filter: LOW|MEDIUM|HIGH|CRITICAL"),
    ward_id: Optional[int] = Query(None, description="Filter by ward ID"),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
):
    """
    Returns a GeoJSON FeatureCollection where each feature is a road segment
    with its current composite flood risk score, class, and DrainWatch data.
    """
    try:
        segments = await get_predictor().get_scored_segments(
            risk_class_filter=risk_class,
            ward_id=ward_id,
            min_score=min_score,
            rainfall_mm=rainfall,
        )

        # Automatically store critical segments in MongoDB
        try:
            from src.db.critical_path_service import persist_critical_segments
            await persist_critical_segments(segments, rainfall_mm=rainfall)
        except Exception as err:
            logger.debug(f"MongoDB background sync notice: {err}")

        features = []
        for seg in segments:
            rc = seg["risk_class"]
            features.append({
                "type": "Feature",
                "geometry": seg["geometry"],
                "properties": {
                    "segment_id": seg["segment_id"],
                    "name": seg.get("name", "Unnamed Road"),
                    "risk_score": round(seg["risk_score"], 3),
                    "risk_class": rc,
                    "color": RISK_COLORS.get(rc, "#94a3b8"),
                    "accessible": seg["accessible"],
                    "predicted_depth_cm": seg.get("predicted_depth_cm"),
                    "rainfall_24h_mm": rainfall,
                    "flood_probability": seg.get("flood_probability"),
                    "elevation_m": seg.get("elevation_m"),
                    "slope_deg": seg.get("slope_deg"),
                    "dist_to_water_m": seg.get("dist_to_water_m"),
                    "dist_to_drain_m": seg.get("dist_to_drain_m"),
                    "cctv_monitored": seg.get("cctv_monitored", False),
                    "cctv_camera_id": seg.get("cctv_camera_id"),
                    "drainwatch_score": seg.get("drainwatch_score"),
                    "explanation": seg.get("explanation", ""),
                    "scored_at": datetime.now(timezone.utc).isoformat(),
                },
            })

        scores = [s["risk_score"] for s in segments]
        total = len(segments)
        accessible_count = sum(1 for s in segments if s["accessible"])
        clearance_pct = round((accessible_count / total) * 100, 1) if total else 100.0

        return JSONResponse({
            "type": "FeatureCollection",
            "features": features,
            "meta": {
                "rainfall_mm": rainfall,
                "total_segments": total,
                "critical": sum(1 for s in segments if s["risk_class"] == "CRITICAL"),
                "high": sum(1 for s in segments if s["risk_class"] == "HIGH"),
                "medium": sum(1 for s in segments if s["risk_class"] == "MEDIUM"),
                "low": sum(1 for s in segments if s["risk_class"] == "LOW"),
                "inaccessible": total - accessible_count,
                "clearance_percentage": clearance_pct,
                "avg_risk": round(sum(scores) / len(scores), 3) if scores else 0,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        })

    except Exception as e:
        logger.exception("Error building risk map")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/layers", summary="Get Chennai GIS environmental layers")
async def get_layers():
    """Returns waterways, waterbodies, drains, and CCTV stations."""
    try:
        from app.data_loader import (
            get_chennai_waterways,
            get_chennai_waterbodies,
            get_chennai_drains,
            get_chennai_cctv_cameras,
        )

        waterways = json.loads(get_chennai_waterways().to_json())
        waterbodies = json.loads(get_chennai_waterbodies().to_json())
        drains = json.loads(get_chennai_drains().to_json())
        cctv = get_chennai_cctv_cameras()

        return {
            "waterways": waterways,
            "waterbodies": waterbodies,
            "drains": drains,
            "cctv_stations": cctv,
        }
    except Exception as e:
        logger.exception("Error loading layers")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk-map/grid", summary="Get sample flood grid cells from trained GeoJSON")
async def get_risk_grid(limit: int = Query(250, ge=10, le=1000)):
    """Returns sample points from trained/chennai_flood_risk.geojson for heatmap visualization."""
    try:
        from pathlib import Path
        geojson_path = Path("trained/chennai_flood_risk.geojson")
        if geojson_path.exists():
            with open(geojson_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            features = data.get("features", [])
            step = max(1, len(features) // limit)
            sampled = features[::step][:limit]
            return {"type": "FeatureCollection", "features": sampled}
        return {"type": "FeatureCollection", "features": []}
    except Exception as e:
        logger.exception(f"Error loading grid GeoJSON: {e}")
        return {"type": "FeatureCollection", "features": []}


@router.get("/stats", summary="Dashboard summary statistics")
async def get_stats():
    segments = await get_predictor().get_scored_segments(rainfall_mm=120.0)
    scores = [s["risk_score"] for s in segments]
    total = len(segments)
    accessible_count = sum(1 for s in segments if s["accessible"])
    return {
        "total_segments": total,
        "critical": sum(1 for s in segments if s["risk_class"] == "CRITICAL"),
        "high": sum(1 for s in segments if s["risk_class"] == "HIGH"),
        "clearance_percentage": round((accessible_count / max(total, 1)) * 100, 1),
        "avg_risk": round(sum(scores) / max(len(scores), 1), 3),
        "monitored_cctv_stations": 7,
    }

