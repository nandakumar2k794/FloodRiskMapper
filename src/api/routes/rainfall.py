"""
Rainfall API — current and historical rainfall readings for Chennai.
"""
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List

from fastapi import APIRouter, HTTPException
from src.ingestion.rainfall_api import RainfallFetcher
from src.db.mongo import get_db

logger = logging.getLogger(__name__)
router = APIRouter()
_fetcher = RainfallFetcher()


@router.get("/rainfall", summary="Current rainfall readings for Chennai stations")
async def get_rainfall():
    """
    Returns rainfall data for all monitored stations in Chennai.
    Data sourced from OpenWeatherMap (live) or last cached reading.
    """
    try:
        data = await _fetcher.get_current()
        return data
    except Exception as e:
        logger.exception("Rainfall fetch error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rainfall/history", summary="Historical rainfall for a station")
async def get_rainfall_history(station_id: str, hours: int = 24):
    """Returns up to 72 hours of hourly rainfall readings from cache."""
    try:
        db = get_db()
        since = datetime.now(timezone.utc) - timedelta(hours=min(hours, 72))
        cursor = db["rainfall_readings"].find(
            {"station_id": station_id, "recorded_at": {"$gte": since}},
            {"_id": 0},
        ).sort("recorded_at", 1)
        readings = await cursor.to_list(length=1000)
        return {"station_id": station_id, "readings": readings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
