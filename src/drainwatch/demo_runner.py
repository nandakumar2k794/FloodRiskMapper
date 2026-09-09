"""
DrainWatch Demo Runner — processes pre-recorded videos through YOLOv8
and stores observations in MongoDB. CPU-optimised for hackathon mode.
"""
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

import cv2

from config.constants import (
    DRAINWATCH_CLASSES, DRAINWATCH_CLASS_SCORES,
    DRAINWATCH_WINDOW_FRAMES, DRAINWATCH_FPS,
)
from src.db.mongo import get_db

logger = logging.getLogger(__name__)


class DrainWatchDemoRunner:
    """
    Runs YOLOv8n inference on video frames and aggregates drain conditions.

    Falls back to a rule-based heuristic if YOLOv8 model is not available,
    ensuring the demo always works regardless of model download state.
    """

    def __init__(self):
        self._model = self._load_yolo()

    def _load_yolo(self):
        try:
            from ultralytics import YOLO
            import os
            model_path = "models/yolov8n_drainwatch.pt"
            if not Path(model_path).exists():
                # Fall back to base YOLOv8n (pretrained COCO)
                logger.warning("Custom YOLO weights not found — using base YOLOv8n")
                model_path = "yolov8n.pt"
            model = YOLO(model_path)
            logger.info(f"✅ YOLOv8 loaded: {model_path}")
            return model
        except Exception as e:
            logger.warning(f"YOLOv8 load failed ({e}) — using heuristic fallback")
            return None

    async def process_video(
        self,
        video_path: str,
        camera_id: str,
        lat: float,
        lon: float,
        location_name: str = "",
    ):
        """Process a video file and store observations in MongoDB."""
        logger.info(f"🎥 Processing video: {video_path} | camera={camera_id}")

        if self._model:
            observations = self._yolo_process(video_path, camera_id, lat, lon, location_name)
        else:
            observations = self._heuristic_process(video_path, camera_id, lat, lon, location_name)

        if observations:
            db = get_db()
            await db["drain_observations"].insert_many(observations)
            logger.info(f"💾 Stored {len(observations)} observations for {camera_id}")

    def _yolo_process(self, video_path, camera_id, lat, lon, location_name):
        """YOLOv8 inference on sampled frames."""
        observations = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_interval = max(1, int(fps / DRAINWATCH_FPS))
        frame_idx = 0
        window = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % frame_interval != 0:
                continue

            results = self._model(frame, verbose=False, imgsz=320)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = r.names.get(cls_id, "unknown")
                    # Map COCO classes to drain classes heuristically
                    drain_cls = self._map_coco_to_drain(class_name, conf)
                    window.append((drain_cls, conf))

            if len(window) >= DRAINWATCH_WINDOW_FRAMES:
                obs = self._aggregate_window(window, camera_id, lat, lon, location_name)
                observations.append(obs)
                window = []

        if window:
            obs = self._aggregate_window(window, camera_id, lat, lon, location_name)
            observations.append(obs)

        cap.release()
        return observations

    def _map_coco_to_drain(self, coco_class: str, conf: float) -> str:
        """Heuristic mapping from COCO classes to DrainWatch classes."""
        mapping = {
            "person": "standing_water_minor",    # people wading = water
            "car": "standing_water_minor",
            "truck": "standing_water_major",
            "boat": "road_waterlogged",
            "toilet": "drain_clear",
            "sink": "drain_clear",
            "bottle": "drain_blocked_garbage",
            "cup": "drain_blocked_garbage",
            "handbag": "drain_blocked_garbage",
            "backpack": "drain_blocked_garbage",
        }
        return mapping.get(coco_class, "drain_clear")

    def _heuristic_process(self, video_path, camera_id, lat, lon, location_name):
        """Rule-based fallback when YOLO is unavailable."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # Completely synthetic demo observation
            return [self._synthetic_observation(camera_id, lat, lon, location_name)]

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
        observations = []
        for i in range(0, total_frames, DRAINWATCH_WINDOW_FRAMES):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            obs = self._analyse_frame_heuristic(frame, camera_id, lat, lon, location_name)
            observations.append(obs)

        cap.release()
        return observations

    def _analyse_frame_heuristic(self, frame, camera_id, lat, lon, location_name):
        """Estimate drain condition from frame brightness/blue channel ratio."""
        import numpy as np
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # High saturation + low value in lower half → possible waterlogging
        lower = hsv[frame.shape[0]//2:, :, :]
        mean_s = float(np.mean(lower[:, :, 1]))
        mean_v = float(np.mean(lower[:, :, 2]))

        # Simple heuristic thresholds
        if mean_v < 60:
            dominant = "standing_water_major"
            blockage = 0.8
        elif mean_s > 100:
            dominant = "drain_blocked_garbage"
            blockage = 0.6
        elif mean_v < 100:
            dominant = "standing_water_minor"
            blockage = 0.3
        else:
            dominant = "drain_clear"
            blockage = 0.05

        score = DRAINWATCH_CLASS_SCORES.get(dominant, 0.0)
        return self._build_observation(
            camera_id, lat, lon, location_name,
            dominant, blockage,
            max(0.0, score - 0.2),
            max(0.0, score - 0.3),
            DRAINWATCH_WINDOW_FRAMES,
        )

    def _aggregate_window(self, window, camera_id, lat, lon, location_name):
        from collections import Counter
        classes = [w[0] for w in window]
        confs = [w[1] for w in window]
        dominant = Counter(classes).most_common(1)[0][0]

        blockage = sum(
            DRAINWATCH_CLASS_SCORES.get(c, 0) * cf
            for c, cf in zip(classes, confs)
            if "blocked" in c or "garbage" in c or "silt" in c
        ) / max(len(window), 1)

        overflow = sum(
            DRAINWATCH_CLASS_SCORES.get(c, 0) * cf
            for c, cf in zip(classes, confs)
            if "overflow" in c
        ) / max(len(window), 1)

        waterlog = sum(
            DRAINWATCH_CLASS_SCORES.get(c, 0) * cf
            for c, cf in zip(classes, confs)
            if "water" in c or "waterlogged" in c
        ) / max(len(window), 1)

        return self._build_observation(
            camera_id, lat, lon, location_name,
            dominant, min(blockage, 1.0), min(overflow, 1.0), min(waterlog, 1.0),
            len(window),
        )

    def _build_observation(self, camera_id, lat, lon, location_name,
                           dominant, blockage, overflow, waterlog, frame_count):
        composite = (blockage * 0.4 + overflow * 0.35 + waterlog * 0.25)
        return {
            "camera_id": camera_id,
            "location_name": location_name,
            "location": {"type": "Point", "coordinates": [lon, lat]},
            "lat": lat,
            "lon": lon,
            "observed_at": datetime.now(timezone.utc),
            "dominant_condition": dominant,
            "blockage_score": round(min(blockage, 1.0), 3),
            "overflow_score": round(min(overflow, 1.0), 3),
            "waterlogging_score": round(min(waterlog, 1.0), 3),
            "composite_drain_score": round(min(composite, 1.0), 3),
            "frame_count": frame_count,
        }

    def _synthetic_observation(self, camera_id, lat, lon, location_name):
        """Fallback fully synthetic observation for demo."""
        dominant = random.choice(DRAINWATCH_CLASSES)
        score = DRAINWATCH_CLASS_SCORES[dominant]
        return self._build_observation(
            camera_id, lat, lon, location_name,
            dominant, score * 0.9, score * 0.6, score * 0.7, 30,
        )

    def seed_demo_cameras(self, cameras: list[dict]):
        """
        Pre-seed MongoDB with synthetic drain observations for demo cameras.
        Call this from the seed script.
        """
        import asyncio
        async def _seed():
            db = get_db()
            docs = []
            for cam in cameras:
                obs = self._synthetic_observation(
                    cam["camera_id"], cam["lat"], cam["lon"], cam.get("name", "")
                )
                docs.append(obs)
            if docs:
                await db["drain_observations"].insert_many(docs)
                logger.info(f"✅ Seeded {len(docs)} demo drain observations")
        asyncio.run(_seed())
