"""
Critical Paths API — Retrieves and manages stored critical road segments and routes in MongoDB.
"""
import logging
from typing import Optional
from fastapi import APIRouter, Query, HTTPException

from src.db.critical_path_service import (
    get_critical_paths,
    get_critical_path_stats,
    persist_critical_segments,
)
from src.ml.predictor import RiskPredictor

logger = logging.getLogger(__name__)
router = APIRouter()
_predictor = RiskPredictor()


@router.get("/critical-paths", summary="Get critical paths and inundated corridors from MongoDB")
async def fetch_critical_paths(
    min_risk: float = Query(0.5, ge=0.0, le=1.0, description="Minimum risk score threshold"),
    limit: int = Query(50, ge=1, le=200),
    status: Optional[str] = Query(None, description="Filter: IMPASSABLE | HAZARDOUS"),
):
    """
    Retrieves stored critical road segments and flood-vulnerable corridors
    directly from MongoDB `critical_paths` collection.
    """
    try:
        paths = await get_critical_paths(min_risk=min_risk, limit=limit, status=status)
        stats = await get_critical_path_stats()
        return {
            "critical_paths": paths,
            "total_returned": len(paths),
            "stats": stats,
        }
    except Exception as e:
        logger.exception("Error fetching critical paths from MongoDB")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/critical-paths/stats", summary="Get MongoDB critical path aggregate statistics")
async def fetch_stats():
    """Returns real-time analytics from MongoDB critical paths collection."""
    try:
        return await get_critical_path_stats()
    except Exception as e:
        logger.exception("Error fetching critical path stats")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/critical-paths/sync", summary="Sync current XGBoost critical segments to MongoDB")
async def sync_critical_paths(
    rainfall: float = Query(140.0, ge=0.0, le=400.0, description="Rainfall scenario mm/24h")
):
    """
    Evaluates citywide road risk with XGBoost model and persists all critical/high-risk
    segments into MongoDB `critical_paths` collection.
    """
    try:
        segments = await _predictor.get_scored_segments(rainfall_mm=rainfall)
        synced_count = await persist_critical_segments(segments, rainfall_mm=rainfall)
        stats = await get_critical_path_stats()
        return {
            "status": "SUCCESS",
            "synced_count": synced_count,
            "rainfall_scenario_mm": rainfall,
            "stats": stats,
            "message": f"Successfully synced {synced_count} critical corridors into MongoDB collection 'critical_paths'.",
        }
    except Exception as e:
        logger.exception("Error syncing critical paths to MongoDB")
        raise HTTPException(status_code=500, detail=str(e))
