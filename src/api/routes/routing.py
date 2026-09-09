"""
Routing API — finds safest/fastest alternative routes avoiding flood-risk roads.
"""
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from src.api.schemas import RouteRequest, RouteResponse
from src.api.app_state import get_finder

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/route", response_model=RouteResponse, summary="Find safe alternative routes")
async def find_route(req: RouteRequest):
    """
    Given an origin and destination, returns up to 3 alternative routes
    ranked by composite flood risk. Inaccessible segments are penalised
    or avoided based on the selected mode.
    """
    try:
        routes = await get_finder().find(
            origin=(req.origin.lat, req.origin.lon),
            destination=(req.destination.lat, req.destination.lon),
            vehicle_type=req.vehicle_type.value,
            mode=req.mode,
        )
        if not routes:
            raise HTTPException(
                status_code=404,
                detail="No route found between the given points.",
            )
        return RouteResponse(
            routes=routes,
            recommended_route_id=routes[0].route_id,
            origin=req.origin,
            destination=req.destination,
            computed_at=datetime.now(timezone.utc),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Routing error")
        raise HTTPException(status_code=500, detail=str(e))
