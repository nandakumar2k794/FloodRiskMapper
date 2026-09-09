"""
Rainfall API fetcher — OpenWeatherMap + synthetic fallback for demo.
"""
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

# Chennai weather stations (lat, lon)
CHENNAI_STATIONS = [
    ("IMD_NUNGAMBAKKAM", "Nungambakkam",    13.0600, 80.2420),
    ("IMD_MEENAMBAKKAM", "Meenambakkam",    12.9900, 80.1800),
    ("IMD_ADYAR",        "Adyar",           13.0050, 80.2570),
    ("IMD_VELACHERY",    "Velachery",       12.9780, 80.2180),
    ("IMD_TONDIARPET",   "Tondiarpet",      13.1210, 80.2900),
    ("IMD_AMBATTUR",     "Ambattur",        13.1120, 80.1610),
    ("IMD_SHOLINGANALLUR","Sholinganallur", 12.9010, 80.2270),
]


class RainfallFetcher:

    async def get_current(self) -> Dict[str, Any]:
        try:
            return await self._fetch_open_meteo()
        except Exception as e:
            logger.warning(f"Open-Meteo fetch failed ({e}) — using synthetic data")
            return self._synthetic_rainfall()

    async def _fetch_open_meteo(self) -> Dict[str, Any]:
        """Fetch from open source Open-Meteo API for each station location."""
        readings = []
        async with httpx.AsyncClient(timeout=10.0) as client:
            tasks = [
                client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "current": "rain,precipitation",
                        "timezone": "auto"
                    },
                )
                for _, _, lat, lon in CHENNAI_STATIONS
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        for (sid, name, lat, lon), resp in zip(CHENNAI_STATIONS, responses):
            if isinstance(resp, Exception):
                continue
            data = resp.json()
            # Open-Meteo returns mm per hour in 'current.rain' or 'current.precipitation'
            current = data.get("current", {})
            rain_1h = current.get("rain", current.get("precipitation", 0.0))
            readings.append({
                "station_id": sid,
                "station_name": name,
                "lat": lat, "lon": lon,
                "rainfall_mm": float(rain_1h),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "source": "open-meteo",
            })

        return self._build_summary(readings)

    def _synthetic_rainfall(self) -> Dict[str, Any]:
        """Synthetic rainfall distribution simulating a heavy rainfall event."""
        import random
        random.seed(int(datetime.now().timestamp()) // 3600)  # changes hourly

        base = random.uniform(20, 80)   # mm/h base
        readings = []
        for sid, name, lat, lon in CHENNAI_STATIONS:
            r = max(0, base + random.gauss(0, 15))
            readings.append({
                "station_id": sid,
                "station_name": name,
                "lat": lat, "lon": lon,
                "rainfall_mm": round(r, 1),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "source": "synthetic_demo",
            })

        return self._build_summary(readings)

    def _build_summary(self, readings: list) -> Dict[str, Any]:
        if not readings:
            return {"stations": [], "city_avg_1h_mm": 0,
                    "city_avg_6h_mm": 0, "city_avg_24h_mm": 0,
                    "imd_alert_level": "NORMAL",
                    "last_updated": datetime.now(timezone.utc).isoformat()}

        avg_1h = sum(r["rainfall_mm"] for r in readings) / len(readings)
        avg_6h = avg_1h * 5.5
        avg_24h = avg_1h * 18

        if avg_24h >= 204.5:
            level = "EXTREMELY HEAVY"
        elif avg_24h >= 115.6:
            level = "VERY HEAVY"
        elif avg_24h >= 64.5:
            level = "HEAVY"
        elif avg_24h >= 15.6:
            level = "MODERATE"
        else:
            level = "LIGHT"

        return {
            "stations": readings,
            "city_avg_1h_mm": round(avg_1h, 1),
            "city_avg_6h_mm": round(avg_6h, 1),
            "city_avg_24h_mm": round(avg_24h, 1),
            "imd_alert_level": level,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
