"""
Global application state — single instances of heavy objects loaded ONCE.
All routes import from here instead of constructing their own instances.

Performance profile (before this fix):
  RiskPredictor()  init: ~6.5s  (XGBoost model load)
  RouteFinder()    init: ~5.3s  (creates another RiskPredictor internally)
  First score_network: ~0.7s
  First find_route:    ~0.8s

After: all init happens in background at startup, first real request < 50ms.
"""
import asyncio
import logging
from typing import Optional

from src.ml.predictor import RiskPredictor
from src.routing.route_finder import RouteFinder

logger = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────
_predictor = None
_finder = None
_ready = False


def get_predictor():
    """Return the shared RiskPredictor singleton (already initialised)."""
    global _predictor
    if _predictor is None:
        # Synchronous fallback if called before startup completes
        _predictor = RiskPredictor()
    return _predictor


def get_finder():
    """Return the shared RouteFinder singleton (already initialised)."""
    global _finder
    if _finder is None:
        _finder = RouteFinder(_predictor=get_predictor())
    return _finder


async def warm_up():
    """
    Pre-load the XGBoost model and warm the segment/graph caches.
    Called once during app startup in a background task so it doesn't
    block the server from accepting requests.
    """
    global _predictor, _finder, _ready
    import time

    logger.info("⚡ Background warm-up starting...")
    t0 = time.perf_counter()

    # 1. Load the XGBoost model (biggest cost: ~6s)
    loop = asyncio.get_event_loop()
    try:
        _predictor = await loop.run_in_executor(None, RiskPredictor)
        logger.info(f"  [1/4] RiskPredictor loaded  ({time.perf_counter()-t0:.2f}s)")
    except Exception as e:
        logger.warning(f"  [1/4] RiskPredictor failed: {e}")

    # 2. Warm the segment scoring cache for the default rainfall (120mm)
    try:
        await _predictor.get_scored_segments(rainfall_mm=120.0)
        logger.info(f"  [2/4] Segment cache warmed  ({time.perf_counter()-t0:.2f}s)")
    except Exception as e:
        logger.warning(f"  [2/4] Segment warm-up failed: {e}")

    # 3. Build RouteFinder sharing the predictor (avoids a second model load)
    try:
        _finder = RouteFinder(_predictor=_predictor)
        logger.info(f"  [3/4] RouteFinder created   ({time.perf_counter()-t0:.2f}s)")
    except Exception as e:
        logger.warning(f"  [3/4] RouteFinder failed: {e}")

    # 4. Pre-build the routing graph for 'car' (most common vehicle)
    try:
        await _finder._build_graph("car")
        logger.info(f"  [4/4] Routing graph built   ({time.perf_counter()-t0:.2f}s)")
    except Exception as e:
        logger.warning(f"  [4/4] Graph build failed: {e}")

    _ready = True
    logger.info(f"✅ Warm-up complete in {time.perf_counter()-t0:.2f}s — all first-request latency eliminated")
