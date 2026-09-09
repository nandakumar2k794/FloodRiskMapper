"""
MongoDB async connection singleton.
"""
import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from config.settings import settings

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


async def connect_mongo():
    global _client, _db
    try:
        _client = AsyncIOMotorClient(settings.mongo_uri, serverSelectionTimeoutMS=2000)
        _db = _client[settings.mongo_db]
        # Quick ping
        await _client.admin.command('ping')
        # Create indexes
        await _db["drain_observations"].create_index(
            [("camera_id", 1), ("observed_at", -1)]
        )
        await _db["drain_observations"].create_index(
            [("location", "2dsphere")]
        )
        await _db["alerts"].create_index([("severity", 1), ("resolved_at", 1)])
        await _db["rainfall_readings"].create_index(
            [("station_id", 1), ("recorded_at", -1)]
        )
        # Critical Paths & Corridors Indexes
        await _db["critical_paths"].create_index(
            [("segment_id", 1), ("recorded_at", -1)]
        )
        await _db["critical_paths"].create_index(
            [("risk_score", -1)]
        )
        await _db["critical_paths"].create_index(
            [("rainfall_mm", 1)]
        )
        await _db["critical_routes"].create_index(
            [("created_at", -1)]
        )
        logger.info(f"✅ Connected to MongoDB: {settings.mongo_db} (Collections: critical_paths, drain_observations, alerts)")
    except Exception as e:
        logger.warning(f"⚠️ MongoDB not available ({e}) — running in fast in-memory store mode")
        _db = None


async def close_mongo():
    global _client
    if _client:
        try:
            _client.close()
        except Exception:
            pass
        logger.info("MongoDB connection closed")


def get_db():
    return _db

