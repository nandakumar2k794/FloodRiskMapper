"""
Critical Paths MongoDB Service — Persists and manages high-risk road corridors,
inundated segments, and flood evacuation routing records in MongoDB.
"""
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from src.db.mongo import get_db

logger = logging.getLogger(__name__)


async def persist_critical_segments(segments: List[Dict[str, Any]], rainfall_mm: float) -> int:
    """
    Saves or updates critical and high-risk road segments in MongoDB collection `critical_paths`.
    Uses upsert on segment_id to maintain the latest status while tracking historical risk timestamps.
    """
    db = get_db()
    if db is None:
        logger.debug("MongoDB not connected; skipping critical path persistence.")
        return 0

    critical_items = [
        s for s in segments
        if s.get("risk_class") in ["CRITICAL", "HIGH"] or s.get("risk_score", 0) >= 0.5
    ]

    if not critical_items:
        return 0

    count = 0
    now = datetime.now(timezone.utc)

    for seg in critical_items:
        doc = {
            "segment_id": seg.get("segment_id"),
            "name": seg.get("name", "Chennai Road Corridor"),
            "risk_score": float(seg.get("risk_score", 0.0)),
            "risk_class": seg.get("risk_class", "CRITICAL"),
            "predicted_depth_cm": int(seg.get("predicted_depth_cm", 0)),
            "accessible": bool(seg.get("accessible", False)),
            "rainfall_mm": float(rainfall_mm),
            "elevation_m": float(seg.get("elevation_m", 0.0)),
            "slope_deg": float(seg.get("slope_deg", 0.0)),
            "dist_to_water_m": float(seg.get("dist_to_water_m", 0.0)),
            "dist_to_drain_m": float(seg.get("dist_to_drain_m", 0.0)),
            "cctv_monitored": bool(seg.get("cctv_monitored", False)),
            "cctv_camera_id": seg.get("cctv_camera_id"),
            "drainwatch_score": float(seg.get("drainwatch_score", 0.0)),
            "explanation": seg.get("explanation", ""),
            "geometry": seg.get("geometry"),
            "last_updated": now,
            "status": "IMPASSABLE" if not seg.get("accessible", False) else "HAZARDOUS",
        }

        try:
            await db["critical_paths"].update_one(
                {"segment_id": seg.get("segment_id")},
                {
                    "$set": doc,
                    "$setOnInsert": {"first_recorded_at": now},
                    "$inc": {"incident_occurrences": 1}
                },
                upsert=True
            )
            count += 1
        except Exception as e:
            logger.warning(f"Error persisting critical path {seg.get('segment_id')}: {e}")

    logger.info(f"💾 Synced {count} critical road corridors to MongoDB 'critical_paths' collection")
    return count


async def get_critical_paths(
    min_risk: float = 0.5,
    limit: int = 50,
    status: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieves persisted critical road corridors from MongoDB."""
    db = get_db()
    if db is None:
        return []

    query: Dict[str, Any] = {"risk_score": {"$gte": min_risk}}
    if status:
        query["status"] = status

    cursor = db["critical_paths"].find(query, {"_id": 0}).sort("risk_score", -1).limit(limit)
    return await cursor.to_list(length=limit)


async def get_critical_path_stats() -> Dict[str, Any]:
    """Computes aggregate analytics over stored critical paths in MongoDB."""
    db = get_db()
    if db is None:
        return {
            "connected": False,
            "total_critical_paths": 0,
            "max_inundation_depth_cm": 0,
            "avg_critical_risk": 0.0,
            "impassable_count": 0,
        }

    try:
        total = await db["critical_paths"].count_documents({})
        impassable = await db["critical_paths"].count_documents({"accessible": False})

        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "avg_risk": {"$avg": "$risk_score"},
                    "max_depth": {"$max": "$predicted_depth_cm"},
                }
            }
        ]
        agg = await db["critical_paths"].aggregate(pipeline).to_list(length=1)
        avg_risk = round(agg[0]["avg_risk"], 3) if agg else 0.0
        max_depth = agg[0]["max_depth"] if agg else 0

        return {
            "connected": True,
            "database": "chennai_flood_risk",
            "collection": "critical_paths",
            "total_critical_paths": total,
            "impassable_count": impassable,
            "avg_critical_risk": avg_risk,
            "max_inundation_depth_cm": max_depth,
            "last_queried": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.warning(f"Error fetching critical path stats from MongoDB: {e}")
        return {
            "connected": True,
            "error": str(e),
            "total_critical_paths": 0,
        }
