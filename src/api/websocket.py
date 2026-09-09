"""
WebSocket endpoint for live risk score and alert push to dashboard.
"""
import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.ml.predictor import RiskPredictor
from src.db.mongo import get_db

logger = logging.getLogger(__name__)
router = APIRouter()
_predictor = RiskPredictor()

# Active connections registry
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        logger.info(f"WS connected. Total: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)
        logger.info(f"WS disconnected. Total: {len(self.active)}")

    async def broadcast(self, data: dict):
        payload = json.dumps(data)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)


manager = ConnectionManager()


@router.websocket("/ws/live")
async def live_feed(ws: WebSocket):
    """
    WebSocket that pushes risk score updates every 30 seconds.
    Payload: { type: "risk_update" | "alert", data: {...} }
    """
    await manager.connect(ws)
    try:
        while True:
            # Push summary stats every 30 seconds
            segments = await _predictor.get_scored_segments()
            total = len(segments)
            payload = {
                "type": "risk_update",
                "data": {
                    "total_segments": total,
                    "critical": sum(1 for s in segments if s["risk_class"] == "CRITICAL"),
                    "high": sum(1 for s in segments if s["risk_class"] == "HIGH"),
                    "inaccessible": sum(1 for s in segments if not s["accessible"]),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            }
            await ws.send_text(json.dumps(payload))

            # Check for new critical alerts
            db = get_db()
            critical_alerts = await db["alerts"].count_documents(
                {"severity": "CRITICAL", "resolved_at": None}
            )
            if critical_alerts > 0:
                await ws.send_text(json.dumps({
                    "type": "alert",
                    "data": {
                        "critical_count": critical_alerts,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                }))

            await asyncio.sleep(30)

    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        logger.exception("WebSocket error")
        manager.disconnect(ws)
