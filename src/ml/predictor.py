"""
Risk Predictor — loads trained ML model and generates per-segment risk scores.
Falls back to physics-based scoring when no trained model is available.
"""
import logging
import pickle
import os
import random
import math
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import numpy as np

from trained.predict import FloodPredictor

from config.constants import (
    RISK_CLASSES, RISK_WEIGHTS, DRAINWATCH_CLASS_SCORES,
    HAND_FLOOD_THRESHOLD_M, TWI_HIGH_THRESHOLD,
)
from config.settings import settings

logger = logging.getLogger(__name__)


def _compute_risk_class(score: float) -> str:
    for cls, (lo, hi) in RISK_CLASSES.items():
        if lo <= score < hi:
            return cls
    return "CRITICAL"


def _rainfall_score(r24: float) -> float:
    """Normalise 24-h rainfall (0–250mm) to [0,1]."""
    return min(r24 / 200.0, 1.0)


def _topo_score(hand_m: float, slope_deg: float, twi: float) -> float:
    """Topographic susceptibility: low HAND + low slope + high TWI = high risk."""
    hand_norm = max(0.0, 1.0 - hand_m / 10.0)
    slope_norm = max(0.0, 1.0 - slope_deg / 15.0)
    twi_norm = min(twi / 12.0, 1.0)
    return (hand_norm * 0.5 + slope_norm * 0.2 + twi_norm * 0.3)


def _proximity_score(dist_river_m: float, dist_lake_m: float) -> float:
    """Closer to water body = higher risk."""
    r = max(0.0, 1.0 - dist_river_m / 500.0)
    l = max(0.0, 1.0 - dist_lake_m / 300.0)
    return max(r, l)


class RiskPredictor:
    """
    Generates flood risk scores for road segments.

    Strategy:
    1. Try to load a trained GBM model from disk.
    2. Fall back to physics-based composite scoring.
    3. Integrate DrainWatch scores from MongoDB when available.
    """

    def __init__(self):
        self._flood_predictor = self._load_flood_predictor()
        self._segment_cache: Dict[float, List[Dict]] = {}
        self._cache_ts: Optional[datetime] = None

    def _load_flood_predictor(self):
        try:
            predictor = FloodPredictor()
            logger.info("✅ Loaded Trained XGBoost Flood Risk Model from trained/final_model.json")
            return predictor
        except Exception as e:
            logger.warning(f"⚠️ Could not load trained/final_model.json ({e}) — fallback to physics scoring")
            return None

    async def get_scored_segments(
        self,
        risk_class_filter: Optional[str] = None,
        ward_id: Optional[int] = None,
        min_score: float = 0.0,
        rainfall_mm: float = 120.0,
    ) -> List[Dict[str, Any]]:
        """Returns scored road segments evaluated dynamically with the trained XGBoost model."""
        cache_key = round(float(rainfall_mm), 1)
        if cache_key in self._segment_cache:
            segments = self._segment_cache[cache_key]
        else:
            segments = self._score_road_network(rainfall_mm)
            self._segment_cache[cache_key] = segments

        # Apply filters
        filtered = segments
        if risk_class_filter:
            filtered = [s for s in filtered if s["risk_class"].upper() == risk_class_filter.upper()]
        if min_score > 0:
            filtered = [s for s in filtered if s["risk_score"] >= min_score]

        return filtered

    def _score_road_network(self, rainfall_mm: float) -> List[Dict[str, Any]]:
        """
        Runs the trained XGBoost model across all Chennai arterial and collector road segments.
        Combines XGBoost flood probability with terrain, hydrology, and DrainWatch CCTV deltas.
        """
        try:
            from app.data_loader import get_chennai_roads_network, get_chennai_cctv_cameras
            roads_gdf = get_chennai_roads_network()
            cctv_list = get_chennai_cctv_cameras()
            cctv_by_segment = {c.get("road_segment_id"): c for c in cctv_list if c.get("road_segment_id")}
        except Exception as e:
            logger.warning(f"Error loading road network from data_loader: {e}")
            return self._generate_demo_segments()

        scored_segments = []
        for _, row in roads_gdf.iterrows():
            geom = row.geometry
            coords = [[round(p[0], 5), round(p[1], 5)] for p in list(geom.coords)]
            seg_id = row.get("segment_id", f"seg_{_}")
            name = row.get("name", "Chennai Corridor")

            elev = float(row.get("elevation_mean", 7.0))
            slope = float(row.get("slope", 1.0))
            dist_river = float(row.get("dist_to_water_m", 300.0))
            dist_drain = float(row.get("dist_to_drain_m", 25.0))
            hist_freq = float(row.get("historical_flood_frequency", 0.1))

            # 1. XGBoost Trained Model Prediction
            # Feature values mapped to training distribution scale:
            # - distance_to_road = 0 because we ARE scoring the road segment itself
            # - distance_to_drainage: metres to nearest drain (5-500m)
            # - distance_to_river: metres to nearest river (20-5000m)
            # - drainage_density: derived from dist_drain inverse
            # - elevation/slope: from segment terrain attributes
            prob = 0.2
            if self._flood_predictor:
                drain_density = max(1.0, min(50.0, 500.0 / max(dist_drain, 5.0)))
                elev_std = max(0.1, slope * 0.3)
                feature_input = {
                    "rainfall": float(rainfall_mm),
                    "distance_to_road": 0.0,          # we ARE the road
                    "distance_to_drainage": float(dist_drain),
                    "distance_to_river": float(dist_river),
                    "drainage_density": drain_density,
                    "land_use": 1,
                    "soil": 1,
                    "mean_elevation": float(elev),
                    "min_elevation": max(0.1, float(elev) - float(slope) * 0.5),
                    "max_elevation": float(elev) + float(slope) * 0.5,
                    "elevation_range": max(0.1, float(slope)),
                    "elevation_std": elev_std,
                    "slope": float(slope),
                }
                try:
                    pred_res = self._flood_predictor.predict(feature_input)
                    prob = float(pred_res.get("flood_probability", 0.2))
                except Exception as e:
                    logger.debug(f"XGBoost predict error: {e}")
                    prob = min(1.0, (rainfall_mm / 250.0) * 0.6 + (1.0 - min(elev / 15.0, 1.0)) * 0.4)
            else:
                prob = min(1.0, (rainfall_mm / 250.0) * 0.6 + (1.0 - min(elev / 15.0, 1.0)) * 0.4)

            # 2. Additive CCTV DrainWatch Adjustment (if monitored by camera)
            cctv = cctv_by_segment.get(seg_id)
            cctv_delta = 0.0
            explanation_points = []

            if rainfall_mm > 150:
                explanation_points.append(f"Extreme rainfall accumulation ({rainfall_mm:.0f} mm/24h)")
            elif rainfall_mm > 70:
                explanation_points.append(f"Heavy monsoon rainfall ({rainfall_mm:.0f} mm/24h)")

            if elev < 6.0:
                explanation_points.append(f"Low-lying terrain ({elev:.1f} m elevation)")
            if dist_river < 150:
                explanation_points.append(f"Proximity to major waterway ({dist_river:.0f} m)")

            if cctv:
                blk = float(cctv.get("blockage_percentage", 0))
                overflow = bool(cctv.get("overflow_detected", False))
                if overflow or blk >= 50:
                    cctv_delta = 0.20 if overflow else 0.12
                    explanation_points.append(f"DrainWatch CCTV: Drain inlet {blk:.0f}% blocked by debris")

            composite_score = min(max(prob + cctv_delta, 0.0), 1.0)

            # Risk classification
            if composite_score < 0.25:
                risk_class = "LOW"
                depth_cm = int(composite_score * 12)
            elif composite_score < 0.50:
                risk_class = "MEDIUM"
                depth_cm = int(10 + (composite_score - 0.25) * 50)
            elif composite_score < 0.75:
                risk_class = "HIGH"
                depth_cm = int(22 + (composite_score - 0.50) * 70)
            else:
                risk_class = "CRITICAL"
                depth_cm = int(40 + (composite_score - 0.75) * 80)

            accessible = depth_cm < 25 and composite_score < 0.80

            if not explanation_points:
                explanation_points.append("Normal drainage capacity and elevation gradient.")

            scored_segments.append({
                "segment_id": seg_id,
                "name": name,
                "risk_score": round(composite_score, 3),
                "risk_class": risk_class,
                "flood_probability": round(prob, 3),
                "accessible": accessible,
                "predicted_depth_cm": depth_cm,
                "rainfall_24h_mm": rainfall_mm,
                "drainwatch_score": round(cctv_delta, 2) if cctv else 0.0,
                "elevation_m": elev,
                "slope_deg": slope,
                "dist_to_water_m": dist_river,
                "dist_to_drain_m": dist_drain,
                "cctv_monitored": cctv is not None,
                "cctv_camera_id": cctv["camera_id"] if cctv else None,
                "explanation": "; ".join(explanation_points),
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                },
            })

        return scored_segments

    def _score_segment(self, seg: Dict) -> float:
        rf = _rainfall_score(seg.get("rainfall_24h_mm", 0))
        topo = _topo_score(
            seg.get("hand_m", 5.0),
            seg.get("slope_deg", 2.0),
            seg.get("twi", 6.0),
        )
        hist = seg.get("hist_flood_freq", 0.0)
        prox = _proximity_score(
            seg.get("dist_river_m", 500),
            seg.get("dist_lake_m", 300),
        )
        dw = seg.get("drainwatch_score", 0.0)

        w = RISK_WEIGHTS
        score = (
            w["rainfall_score"]    * rf +
            w["topographic_score"] * topo +
            w["historical_score"]  * hist +
            w["drainwatch_score"]  * dw +
            w["proximity_score"]   * prox
        )
        return round(min(max(score, 0.0), 1.0), 4)

    def _generate_demo_segments(self) -> List[Dict]:
        """
        Synthetic road network for Chennai covering major corridors.
        Generates realistic risk distribution — uses XGBoost model when available,
        falls back to physics heuristic. Features are mapped to training scale.
        """
        random.seed(42)
        roads = [
            # (name, lat1, lon1, lat2, lon2, base_risk, near_water, elev_m, slope_deg, dist_river_m, dist_drain_m)
            ("Anna Salai",          13.0450, 80.2500, 13.0700, 80.2600, 0.15, False,  9.0, 2.0,  600, 80),
            ("Rajaji Salai",        13.0850, 80.2850, 13.0920, 80.2920, 0.25, True,   5.0, 0.8,  120, 30),
            ("Poonamallee High Rd", 13.0400, 80.1700, 13.0600, 80.1900, 0.20, False, 10.0, 1.5,  800, 60),
            ("GST Road",            12.9200, 80.1500, 13.0000, 80.1800, 0.35, False,  8.0, 1.2,  500, 50),
            ("Adyar Bridge",        13.0010, 80.2560, 13.0060, 80.2610, 0.88, True,   2.5, 0.3,   25, 10),
            ("Kotturpuram Link",    13.0100, 80.2450, 13.0150, 80.2500, 0.75, True,   3.5, 0.5,   60, 15),
            ("Velachery Main Rd",   12.9780, 80.2180, 12.9900, 80.2280, 0.82, True,   3.0, 0.4,   45, 12),
            ("Saidapet Underpass",  13.0200, 80.2320, 13.0230, 80.2350, 0.91, True,   2.0, 0.2,   30,  8),
            ("T.Nagar North",       13.0390, 80.2320, 13.0500, 80.2360, 0.40, False,  8.0, 1.0,  350, 45),
            ("Porur Junction",      13.0360, 80.1580, 13.0410, 80.1630, 0.55, False,  7.0, 1.5,  400, 55),
            ("Ambattur Estate",     13.1100, 80.1580, 13.1200, 80.1680, 0.45, False, 10.0, 2.0,  700, 70),
            ("Kolathur Main",       13.1200, 80.2100, 13.1300, 80.2200, 0.30, False, 11.0, 2.5,  900, 90),
            ("Tondiarpet High St",  13.1180, 80.2860, 13.1280, 80.2930, 0.60, True,   4.0, 0.6,   80, 20),
            ("Mylapore Tank Rd",    13.0330, 80.2680, 13.0380, 80.2730, 0.70, True,   4.5, 0.7,  100, 25),
            ("Perungudi OMR",       12.9600, 80.2420, 12.9750, 80.2520, 0.65, True,   5.0, 0.8,  150, 30),
            ("Pallavaram Bridge",   12.9680, 80.1500, 12.9740, 80.1560, 0.78, True,   3.0, 0.4,   40, 10),
            ("Avadi Main Road",     13.1140, 80.1060, 13.1260, 80.1160, 0.22, False, 12.0, 3.0, 1200, 100),
            ("Thiruvanmiyur Rd",    12.9850, 80.2560, 12.9950, 80.2620, 0.50, True,   5.5, 0.9,  180, 35),
            ("Medavakkam Tank",     12.9270, 80.1930, 12.9380, 80.2000, 0.80, True,   2.5, 0.3,   35,  8),
            ("Koyambedu Jct",       13.0720, 80.2020, 13.0790, 80.2080, 0.38, False,  9.0, 1.8,  600, 65),
            ("Ennore Port Rd",      13.2100, 80.3150, 13.2200, 80.3220, 0.68, True,   3.5, 0.5,   70, 18),
            ("Egmore Main",         13.0780, 80.2640, 13.0850, 80.2700, 0.28, False,  9.5, 1.5,  550, 60),
            ("Madhavaram High Rd",  13.1490, 80.2340, 13.1590, 80.2410, 0.35, False, 11.0, 2.0,  800, 80),
            ("Sholinganallur Rd",   12.9010, 80.2250, 12.9110, 80.2320, 0.62, True,   4.5, 0.7,  130, 28),
            ("Nungambakkam High",   13.0590, 80.2420, 13.0660, 80.2490, 0.18, False, 11.0, 2.5, 1000, 95),
        ]

        current_rainfall_24h = 120.0  # Fixed for reproducible graph building

        segments = []
        for i, (name, lat1, lon1, lat2, lon2, base_risk, near_water, elev, slope, dist_river, dist_drain) in enumerate(roads):
            # XGBoost prediction with correct feature scale
            prob = base_risk  # default if model unavailable
            if self._flood_predictor:
                drain_density = max(1.0, min(50.0, 500.0 / max(dist_drain, 5.0)))
                elev_std = max(0.1, slope * 0.3)
                feature_input = {
                    "rainfall": current_rainfall_24h,
                    "distance_to_road": 0.0,          # we ARE the road
                    "distance_to_drainage": float(dist_drain),
                    "distance_to_river": float(dist_river),
                    "drainage_density": drain_density,
                    "land_use": 2 if near_water else 1,
                    "soil": 2 if near_water else 1,
                    "mean_elevation": float(elev),
                    "min_elevation": max(0.1, float(elev) - slope * 0.5),
                    "max_elevation": float(elev) + slope * 0.5,
                    "elevation_range": max(0.1, slope),
                    "elevation_std": elev_std,
                    "slope": float(slope),
                }
                try:
                    pred_res = self._flood_predictor.predict(feature_input)
                    prob = float(pred_res.get("flood_probability", base_risk))
                except Exception:
                    prob = base_risk
            else:
                rainfall_factor = _rainfall_score(current_rainfall_24h)
                water_factor = 0.2 if near_water else 0.0
                noise = random.gauss(0, 0.03)
                prob = min(max(base_risk + rainfall_factor * 0.3 + water_factor + noise, 0.0), 1.0)

            score = prob
            depth = score * 60 if near_water else score * 20
            rc = _compute_risk_class(score)
            dw_score = min(score + random.gauss(0, 0.05), 1.0) if near_water else score * 0.5

            segments.append({
                "segment_id": 10_000 + i,
                "osm_id": 100_000_000 + i,
                "name": name,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [lon1, lat1],
                        [(lon1 + lon2) / 2, (lat1 + lat2) / 2],
                        [lon2, lat2],
                    ],
                },
                "length_m": round(
                    math.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) * 111_000, 0
                ),
                "road_type": "primary" if base_risk < 0.4 else "secondary",
                "risk_score": round(score, 3),
                "risk_class": rc,
                "accessible": depth < 25,
                "predicted_depth_cm": round(depth, 1),
                "rainfall_24h_mm": round(current_rainfall_24h, 1),
                "drainwatch_score": round(dw_score, 3),
                "hist_flood_freq": round(base_risk * 0.8, 3),
                "hand_m": round(max(0.5, 5 - score * 4), 2),
                "slope_deg": slope,
                "dist_river_m": dist_river,
                "elevation_m": elev,
                "slope_deg": slope,
                "dist_to_water_m": dist_river,
                "dist_to_drain_m": dist_drain,
                "ward_id": (i % 15) + 1,
                "scored_at": datetime.now(timezone.utc),
            })

        return segments
