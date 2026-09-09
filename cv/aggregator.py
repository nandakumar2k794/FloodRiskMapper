"""
Temporal Observation Aggregator for DrainWatch AI.
Aggregates frame observations across sliding temporal windows to eliminate transient noise
and produce robust, persistent condition intelligence for the ML risk model.

Key features:
- Sliding window buffering per camera_id
- Persistence voting (garbage/overflow require ≥20% of frames)
- Confidence-weighted condition scoring
- Water-rise rate: Δ(water_coverage_%) per minute — relative proxy, NOT absolute depth
- Fallback to unweighted mean when all detection confidences are zero
"""
from typing import Dict, Any, List, Optional
from collections import deque, Counter
from datetime import datetime
import numpy as np

from cv.models import FrameObservation, AggregatedObservation
from config.constants import DRAINWATCH_WINDOW_FRAMES


class TemporalObservationAggregator:
    """
    Sliding window aggregator for camera frame observations.
    """

    def __init__(self, window_size: int = DRAINWATCH_WINDOW_FRAMES) -> None:
        self.window_size = max(1, window_size)
        self._buffers: Dict[str, deque] = {}

    def add_observation(self, observation: FrameObservation) -> None:
        """Add a single frame observation to the camera's sliding window."""
        cam_id = observation.camera_id
        if cam_id not in self._buffers:
            self._buffers[cam_id] = deque(maxlen=self.window_size)
        self._buffers[cam_id].append(observation)

    def get_aggregated_condition(
        self,
        camera_id: str,
        as_of_time: Optional[datetime] = None,
    ) -> AggregatedObservation:
        """
        Compute aggregated metrics across buffered frames for a camera.
        """
        now = as_of_time or datetime.now()
        buf = self._buffers.get(camera_id, deque())

        if not buf:
            return _empty_observation(camera_id, now)

        frames = list(buf)
        return _compute_aggregation(frames, camera_id, now)

    def aggregate_frame_list(
        self,
        frames: List[FrameObservation],
        camera_id: Optional[str] = None,
    ) -> AggregatedObservation:
        """
        One-shot aggregation of a fixed list of frame observations (e.g. from video file processing).
        """
        if not frames:
            cid = camera_id or "camera_01"
            now = datetime.now()
            return _empty_observation(cid, now)

        cid = camera_id or frames[0].camera_id
        now = frames[-1].timestamp
        return _compute_aggregation(frames, cid, now)


def _empty_observation(camera_id: str, now: datetime) -> AggregatedObservation:
    """Return a neutral 'no data' observation used when the camera buffer is empty."""
    return AggregatedObservation(
        camera_id=camera_id,
        start_time=now,
        end_time=now,
        total_frames_analyzed=0,
        garbage_detected=False,
        garbage_area_mean=0.0,
        drain_area_mean=0.0,
        blockage_percentage_mean=0.0,
        water_coverage_percentage_mean=0.0,
        overflow_detected=False,
        confidence_mean=0.0,
        dominant_condition_class="drain_clear",
        condition_score_mean=0.0,
        staleness_hours=999.0,
        is_valid=False,
        water_rise_rate_pct_per_min=0.0,
        condition_score_confidence_weighted=0.0,
        observation_window_minutes=0.0,
        overflow_frame_count=0,
    )


def _compute_aggregation(
    frames: List[FrameObservation],
    camera_id: str,
    as_of_time: datetime,
) -> AggregatedObservation:
    """
    Core aggregation logic shared between sliding-window and one-shot paths.

    Water-rise rate calculation:
    - Computed as a linear regression slope of water_coverage_percentage vs time.
    - Units: percentage-points per minute.
    - This is a RELATIVE proxy for drainage capacity; do not interpret as cm/min.
    - Requires at least 2 frames with distinct timestamps to compute.
    """
    n = len(frames)
    start_time = frames[0].timestamp
    end_time = frames[-1].timestamp

    # Staleness from the most recent frame
    time_diff_sec = (as_of_time - end_time).total_seconds()
    staleness_hours = max(0.0, time_diff_sec / 3600.0)

    # Observation window span
    window_span_sec = max(0.0, (end_time - start_time).total_seconds())
    observation_window_minutes = window_span_sec / 60.0

    # Persistence voting
    garbage_votes = sum(1 for f in frames if f.garbage_detected)
    overflow_votes = sum(1 for f in frames if f.overflow_detected)
    persistent_garbage = (garbage_votes / n) >= 0.20
    persistent_overflow = (overflow_votes / n) >= 0.20

    # Simple means
    mean_g_area = float(np.mean([f.garbage_area for f in frames]))
    mean_d_area = float(np.mean([f.drain_area for f in frames]))
    mean_blockage = float(np.mean([f.blockage_percentage for f in frames]))
    mean_water = float(np.mean([f.water_coverage_percentage for f in frames]))
    mean_conf = float(np.mean([f.confidence for f in frames]))
    mean_score = float(np.mean([f.condition_score for f in frames]))

    # Confidence-weighted condition score
    confs = np.array([f.confidence for f in frames], dtype=float)
    scores = np.array([f.condition_score for f in frames], dtype=float)
    total_conf = confs.sum()
    if total_conf > 1e-9:
        conf_weighted_score = float(np.dot(confs, scores) / total_conf)
    else:
        # All confidences zero (e.g. model unavailable): fall back to unweighted mean
        conf_weighted_score = mean_score

    # Dominant condition class by frame majority vote
    class_counts = Counter(f.condition_class for f in frames)
    dominant_class = class_counts.most_common(1)[0][0]

    # Water-rise rate: Δ(water_coverage_%) per minute via linear regression on timestamps
    water_rise_rate = 0.0
    if n >= 2:
        t0 = frames[0].timestamp
        times_min = np.array(
            [(f.timestamp - t0).total_seconds() / 60.0 for f in frames], dtype=float
        )
        water_vals = np.array([f.water_coverage_percentage for f in frames], dtype=float)
        # Only compute if time range is > 0
        time_range = times_min[-1] - times_min[0]
        if time_range > 1e-6:
            # np.polyfit degree-1 slope
            slope, _ = np.polyfit(times_min, water_vals, 1)
            water_rise_rate = float(slope)

    return AggregatedObservation(
        camera_id=camera_id,
        start_time=start_time,
        end_time=end_time,
        total_frames_analyzed=n,
        garbage_detected=persistent_garbage,
        garbage_area_mean=round(mean_g_area, 1),
        drain_area_mean=round(mean_d_area, 1),
        blockage_percentage_mean=round(mean_blockage, 1),
        water_coverage_percentage_mean=round(mean_water, 1),
        overflow_detected=persistent_overflow,
        confidence_mean=round(mean_conf, 3),
        dominant_condition_class=dominant_class,
        condition_score_mean=round(mean_score, 3),
        staleness_hours=round(staleness_hours, 2),
        is_valid=True,
        water_rise_rate_pct_per_min=round(water_rise_rate, 4),
        condition_score_confidence_weighted=round(conf_weighted_score, 3),
        observation_window_minutes=round(observation_window_minutes, 2),
        overflow_frame_count=overflow_votes,
    )
