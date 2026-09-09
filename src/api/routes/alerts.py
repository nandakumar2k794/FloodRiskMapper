"""
Alerts API — active flood/drain alerts ranked by severity.
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from src.db.mongo import get_db
from config.constants import RISK_COLORS

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/alerts", summary="Get active flood and drain alerts")
async def get_alerts(
    severity: Optional[str] = Query(None, description="Filter: INFO|WARNING|HIGH|CRITICAL"),
    limit: int = Query(50, ge=1, le=200),
):
    """Returns active (unresolved) alerts ordered by severity and creation time."""
    try:
        db = get_db()
        query: dict = {"resolved_at": None}
        if severity:
            query["severity"] = severity.upper()

        cursor = db["alerts"].find(query, {"_id": 0}).sort(
            [("severity_order", -1), ("created_at", -1)]
        ).limit(limit)

        alerts = await cursor.to_list(length=limit)

        severity_rank = {"CRITICAL": 4, "HIGH": 3, "WARNING": 2, "INFO": 1}

        return {
            "alerts": alerts,
            "total": len(alerts),
            "critical_count": sum(1 for a in alerts if a.get("severity") == "CRITICAL"),
            "high_count": sum(1 for a in alerts if a.get("severity") == "HIGH"),
        }
    except Exception as e:
        logger.exception("Error fetching alerts")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/resolve", summary="Mark an alert as resolved")
async def resolve_alert(alert_id: str):
    db = get_db()
    result = await db["alerts"].update_one(
        {"alert_id": alert_id},
        {"$set": {"resolved_at": datetime.now(timezone.utc)}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"status": "resolved", "alert_id": alert_id}
