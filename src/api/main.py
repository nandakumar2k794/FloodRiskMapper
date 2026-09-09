"""
FastAPI application entry point.
Chennai Flood Access Risk Mapper + DrainWatch AI
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config.settings import settings
from src.api.routes import risk_map, routing, alerts, drainwatch, rainfall, critical_paths
from src.api import websocket as ws_module
from src.db.mongo import connect_mongo, close_mongo
from src.api import app_state

logging.basicConfig(
    level=logging.DEBUG if settings.app_debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("🚀 Starting Chennai Flood Access Risk Mapper API")
    await connect_mongo()
    # Fire-and-forget: warm up XGBoost model + routing graph in background.
    # Server is immediately ready for requests; heavy init happens async.
    asyncio.create_task(app_state.warm_up())
    yield
    logger.info("🛑 Shutting down API")
    await close_mongo()


app = FastAPI(
    title="Chennai Flood Access Risk Mapper",
    description=(
        "Road-level flood accessibility scoring and safer alternative routing "
        "for Chennai, augmented by DrainWatch AI visual drainage intelligence."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Routers ─────────────────────────────────────────────────────────────────
app.include_router(risk_map.router,       prefix="/api/v1", tags=["Risk Map"])
app.include_router(routing.router,        prefix="/api/v1", tags=["Routing"])
app.include_router(critical_paths.router, prefix="/api/v1", tags=["Critical Paths"])
app.include_router(alerts.router,         prefix="/api/v1", tags=["Alerts"])
app.include_router(drainwatch.router,     prefix="/api/v1", tags=["DrainWatch"])
app.include_router(rainfall.router,       prefix="/api/v1", tags=["Rainfall"])
app.include_router(ws_module.router,      tags=["WebSocket"])


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "service": "Chennai Flood Access Risk Mapper"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
    )
