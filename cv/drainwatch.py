"""
DrainWatch AI: Continuous Video & Image Analysis Service.
Coordinates video frame extraction, YOLO inference, ROI filtering, and temporal aggregation.

Supported input types:
- Single image (file path, numpy ndarray, PIL Image)
- MP4 / AVI video files
- Local USB/integrated webcams (device index)
- Authorized RTSP / HTTP streams — see process_camera_stream() for disclaimer

IMPORTANT — CCTV access policy:
This system does NOT assume access to Chennai municipal, police, or government CCTV
infrastructure. The RTSP/webcam interfaces are designed for:
  - Operator-owned cameras deployed by the user/organization
  - Field cameras authorized by the data owner
  - Research/hackathon demonstration environments

Integrating third-party camera feeds requires explicit authorization from the feed owner.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image

from cv.models import FrameObservation, AggregatedObservation
from cv.preprocessing import VideoFrameExtractor, ImagePreprocessor, ROISelector, PIIAnonymizer
from cv.detector import DrainDetector
from cv.aggregator import TemporalObservationAggregator, _empty_observation


class DrainWatchService:
    """
    End-to-end DrainWatch AI vision inspection service.

    Analyzes street-level camera feeds for garbage blockages, overflowing drains,
    and road waterlogging. Acts as the coordinator between frame extraction, YOLO
    inference, ROI filtering, and temporal sliding-window aggregation.

    Missing CCTV fallback:
    When a camera has no valid observations (is_valid=False), export_for_ml() returns
    neutral DrainWatch feature values so the Stage 2 ML model falls back gracefully
    to rainfall + terrain + historical features only.
    """

    def __init__(
        self,
        detector: Optional[DrainDetector] = None,
        aggregator: Optional[TemporalObservationAggregator] = None,
        default_fps: float = 1.0,
        enable_pii_anonymization: bool = True,
    ) -> None:
        self.detector = detector or DrainDetector()
        self.aggregator = aggregator or TemporalObservationAggregator()
        self.default_fps = default_fps
        self.camera_rois: Dict[str, ROISelector] = {}
        self.anonymizer = PIIAnonymizer() if enable_pii_anonymization else None

    # ── ROI Management ────────────────────────────────────────────────────────

    def set_camera_roi(
        self,
        camera_id: str,
        roi_box: Tuple[float, float, float, float],
        is_normalized: bool = False,
    ) -> None:
        """
        Assign a specific drain Region of Interest (ROI) box for a camera.

        Parameters:
            camera_id: Unique camera identifier.
            roi_box: (x1, y1, x2, y2) bounding box.
            is_normalized: If True, coordinates are in [0.0, 1.0] relative to frame size.
        """
        self.camera_rois[camera_id] = ROISelector(roi_box=roi_box, is_normalized=is_normalized)

    def clear_camera_roi(self, camera_id: str) -> None:
        """Remove ROI restriction for a camera (full frame becomes active)."""
        self.camera_rois.pop(camera_id, None)

    # ── Image Processing ──────────────────────────────────────────────────────

    def process_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        camera_id: str = "camera_01",
        timestamp: Optional[datetime] = None,
        roi: Optional[ROISelector] = None,
    ) -> FrameObservation:
        """
        Process a single image frame and update the camera's temporal buffer.

        Parameters:
            image_input: Image file path, numpy BGR array, or PIL Image.
            camera_id: Unique camera identifier used for per-camera temporal buffering.
            timestamp: Frame timestamp (defaults to datetime.now()).
            roi: Override ROI; if None, uses camera-registered ROI if available.

        Returns:
            FrameObservation with structured drain condition metrics.
        """
        active_roi = roi or self.camera_rois.get(camera_id)
        
        if self.anonymizer:
            frame_arr = ImagePreprocessor.to_bgr_array(image_input)
            image_input = self.anonymizer.apply_anonymization(frame_arr, active_roi)
            
        obs = self.detector.process_frame(
            frame_input=image_input,
            camera_id=camera_id,
            timestamp=timestamp,
            frame_index=0,
            roi=active_roi,
        )
        self.aggregator.add_observation(obs)
        return obs

    # ── Video Processing ──────────────────────────────────────────────────────

    def process_video(
        self,
        video_path: Union[str, Path],
        camera_id: str = "camera_01",
        fps: Optional[float] = None,
        roi: Optional[ROISelector] = None,
        max_frames: Optional[int] = None,
        start_time: Optional[datetime] = None,
    ) -> AggregatedObservation:
        """
        Extract frames from an MP4/AVI video file, run YOLO inference per frame,
        and aggregate metrics across the full clip.

        Parameters:
            video_path: Path to the video file.
            camera_id: Unique camera identifier.
            fps: Target frame sampling rate (default: self.default_fps).
            roi: Optional ROI override.
            max_frames: Cap on extracted frames.
            start_time: Timestamp for first frame; defaults to datetime.now().

        Returns:
            AggregatedObservation summarising the entire video clip.
        """
        active_fps = fps if fps is not None else self.default_fps
        extractor = VideoFrameExtractor(target_fps=active_fps)
        active_roi = roi or self.camera_rois.get(camera_id)

        frame_observations: List[FrameObservation] = []

        for frame_idx, ts, frame in extractor.extract_frames(
            video_path, max_frames=max_frames, start_time=start_time
        ):
            if self.anonymizer:
                frame = self.anonymizer.apply_anonymization(frame, active_roi)
                
            obs = self.detector.process_frame(
                frame_input=frame,
                camera_id=camera_id,
                timestamp=ts,
                frame_index=frame_idx,
                roi=active_roi,
            )
            self.aggregator.add_observation(obs)
            frame_observations.append(obs)

        return self.aggregator.aggregate_frame_list(frame_observations, camera_id=camera_id)

    # ── Webcam Processing ─────────────────────────────────────────────────────

    def process_webcam(
        self,
        camera_id: str = "camera_01",
        device_index: int = 0,
        fps: Optional[float] = None,
        max_frames: int = 30,
        roi: Optional[ROISelector] = None,
    ) -> AggregatedObservation:
        """
        Capture and analyze frames from a local USB/integrated webcam.

        This method accesses only locally attached cameras (device index).
        It does NOT connect to any external network streams.

        Parameters:
            camera_id: Logical identifier for this camera.
            device_index: OpenCV VideoCapture device index (0 = default webcam).
            fps: Target sampling rate.
            max_frames: Number of frames to analyze before returning.
            roi: Optional ROI override.

        Returns:
            AggregatedObservation across all captured frames.
        """
        return self.process_video(
            video_path=str(device_index),
            camera_id=camera_id,
            fps=fps,
            max_frames=max_frames,
            roi=roi,
        )

    # ── RTSP / HTTP Stream Processing ─────────────────────────────────────────

    def process_camera_stream(
        self,
        camera_id: str,
        stream_url: str,
        fps: Optional[float] = None,
        max_frames: int = 30,
        roi: Optional[ROISelector] = None,
    ) -> AggregatedObservation:
        """
        Ingest and analyze frames from an AUTHORIZED RTSP / HTTP camera stream.

        AUTHORIZATION REQUIRED:
        - This method must only be called with streams the caller is explicitly
          authorized to access.
        - This system does NOT connect to Chennai municipal, police, or government
          CCTV without explicit written authorization from the stream owner.
        - For hackathon demonstrations, use local video files or authorized field cameras.

        Parameters:
            camera_id: Unique camera identifier.
            stream_url: Full RTSP or HTTP stream URL (e.g. "rtsp://192.168.1.10:554/stream").
            fps: Target frame sampling rate.
            max_frames: Number of frames before returning (prevents indefinite streaming).
            roi: Optional ROI override.

        Returns:
            AggregatedObservation across analyzed frames.
        """
        if not stream_url.startswith(("rtsp://", "rtsps://", "http://", "https://", "0", "1", "2")):
            raise ValueError(
                f"Invalid or unsupported stream URL scheme: {stream_url!r}. "
                "Only rtsp://, rtsps://, http://, and https:// streams are supported."
            )

        return self.process_video(
            video_path=stream_url,
            camera_id=camera_id,
            fps=fps,
            max_frames=max_frames,
            roi=roi,
        )

    # ── Query & Export ────────────────────────────────────────────────────────

    def get_latest_condition(
        self,
        camera_id: str,
        as_of_time: Optional[datetime] = None,
    ) -> AggregatedObservation:
        """
        Fetch latest aggregated drainage condition for a camera location.

        Returns an observation with is_valid=False if no data has been buffered yet.
        """
        return self.aggregator.get_aggregated_condition(camera_id, as_of_time=as_of_time)

    def export_for_ml(self, camera_id: str) -> Dict[str, Any]:
        """
        Export vision metrics in the schema expected by Stage 2 ML risk score adjustment.

        When no valid CCTV data is available (is_valid=False), returns neutral/zero values
        with drainwatch_available=False so the ML model falls back to rainfall + terrain
        + historical features only. This prevents CCTV absence from creating missing-value
        artifacts in the feature matrix.
        """
        agg = self.get_latest_condition(camera_id)
        return agg.to_ml_features()

    def list_active_cameras(self) -> List[str]:
        """Return list of camera IDs that currently have buffered observations."""
        return list(self.aggregator._buffers.keys())

    def clear_camera_buffer(self, camera_id: str) -> None:
        """Reset the observation buffer for a camera (e.g. after a session ends)."""
        self.aggregator._buffers.pop(camera_id, None)
