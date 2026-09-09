"""
Video Frame Extraction, Image Preprocessing, and ROI Selection Module.
Supports:
- Image (file path, numpy ndarray, PIL Image)
- MP4 / AVI video files
- Local USB/Integrated Webcams
- Authorized RTSP / HTTP video streams
"""
from typing import Dict, Any, List, Optional, Tuple, Union, Generator
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import cv2
from PIL import Image


class ImagePreprocessor:
    """
    Applies standardization, contrast enhancement, and formatting to camera frames.
    """

    @staticmethod
    def to_bgr_array(image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Convert diverse image inputs into a standard OpenCV BGR numpy array (H, W, 3).
        """
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f"Failed to decode image at: {path}")
            return frame

        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input)
            if len(rgb.shape) == 2:  # grayscale
                return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
            elif rgb.shape[2] == 4:  # RGBA
                return cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            return image_input

        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    @staticmethod
    def enhance_contrast(frame_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on luminance channel
        to improve detection under rainy, murky, or low-light conditions.
        """
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def resize_frame(frame_bgr: np.ndarray, target_size: Tuple[int, int] = (640, 640), keep_aspect_ratio: bool = True) -> np.ndarray:
        """
        Resize frame to target dimensions. If keep_aspect_ratio is True, applies letterboxing.
        """
        if not keep_aspect_ratio:
            return cv2.resize(frame_bgr, target_size, interpolation=cv2.INTER_LINEAR)

        h, w = frame_bgr.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # Center resized image in canvas
        dx = (target_w - nw) // 2
        dy = (target_h - nh) // 2
        canvas[dy : dy + nh, dx : dx + nw] = resized
        return canvas


class PIIAnonymizer:
    """
    Lightweight PII anonymization to comply with privacy policies.
    Blurs regions outside the designated ROI to obscure faces and license plates
    before YOLO detection and frame storage.
    """

    def __init__(self, blur_kernel: Tuple[int, int] = (51, 51)) -> None:
        self.blur_kernel = blur_kernel

    def apply_anonymization(self, frame_bgr: np.ndarray, roi_selector: Optional['ROISelector'] = None) -> np.ndarray:
        """
        Apply heavy Gaussian blur to the frame outside the ROI.
        If no ROI is configured, blurs the top 30% of the image (common face/plate area).
        """
        h, w = frame_bgr.shape[:2]
        
        if roi_selector is None or roi_selector.roi_box is None:
            anonymized = frame_bgr.copy()
            top_h = int(h * 0.3)
            if top_h > 0:
                top_slice = anonymized[0:top_h, 0:w]
                anonymized[0:top_h, 0:w] = cv2.GaussianBlur(top_slice, self.blur_kernel, 0)
            return anonymized

        abs_box = roi_selector.get_absolute_box(frame_bgr.shape)
        if abs_box is None:
            return frame_bgr

        rx1, ry1, rx2, ry2 = map(int, abs_box)
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(w, rx2), min(h, ry2)

        blurred_bg = cv2.GaussianBlur(frame_bgr, self.blur_kernel, 0)
        
        # Paste the sharp ROI back on top
        blurred_bg[ry1:ry2, rx1:rx2] = frame_bgr[ry1:ry2, rx1:rx2]
        return blurred_bg


class ROISelector:
    """
    Region of Interest (ROI) filter for focusing visual analysis on drain inlets / channels.
    """

    def __init__(self, roi_box: Optional[Tuple[float, float, float, float]] = None, is_normalized: bool = False) -> None:
        """
        roi_box: (x1, y1, x2, y2) bounding box defining the drain inspection region.
        is_normalized: If True, coords are in [0.0, 1.0] relative to frame dimensions.
        """
        self.roi_box = roi_box
        self.is_normalized = is_normalized

    def get_absolute_box(self, frame_shape: Tuple[int, ...]) -> Optional[Tuple[float, float, float, float]]:
        """Get absolute pixel coordinates (x1, y1, x2, y2) for the given frame shape."""
        if not self.roi_box:
            return None

        h, w = frame_shape[:2]
        x1, y1, x2, y2 = self.roi_box

        if self.is_normalized:
            return (x1 * w, y1 * h, x2 * w, y2 * h)
        return (float(x1), float(y1), float(x2), float(y2))

    def is_box_in_roi(self, box: Tuple[float, float, float, float], frame_shape: Tuple[int, ...]) -> bool:
        """
        Check if a detection bounding box center or significant portion lies inside the ROI.
        """
        abs_roi = self.get_absolute_box(frame_shape)
        if abs_roi is None:
            return True  # No ROI configured -> whole frame is active

        rx1, ry1, rx2, ry2 = abs_roi
        bx1, by1, bx2, by2 = box

        # Box center
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0

        return bool(rx1 <= cx <= rx2 and ry1 <= cy <= ry2)

    def compute_overlap_area(self, box: Tuple[float, float, float, float], frame_shape: Tuple[int, ...]) -> float:
        """
        Compute pixel intersection area between a detection box and the ROI.
        """
        abs_roi = self.get_absolute_box(frame_shape)
        if abs_roi is None:
            # Whole box area
            return max(0.0, (box[2] - box[0]) * (box[3] - box[1]))

        rx1, ry1, rx2, ry2 = abs_roi
        bx1, by1, bx2, by2 = box

        ix1 = max(rx1, bx1)
        iy1 = max(ry1, by1)
        ix2 = min(rx2, bx2)
        iy2 = min(ry2, by2)

        if ix2 > ix1 and iy2 > iy1:
            return float((ix2 - ix1) * (iy2 - iy1))
        return 0.0


class VideoFrameExtractor:
    """
    Extracts frames from video files, RTSP streams, webcams, or single images
    at a configurable sampling frame rate.
    """

    def __init__(self, target_fps: float = 1.0) -> None:
        """
        target_fps: Number of frames to sample per second (e.g. 1.0 = 1 fps).
        """
        self.target_fps = max(0.1, target_fps)

    def extract_frames(
        self,
        source: Union[str, Path, int, np.ndarray, Image.Image],
        max_frames: Optional[int] = None,
        start_time: Optional[datetime] = None,
    ) -> Generator[Tuple[int, datetime, np.ndarray], None, None]:
        """
        Generator yielding (frame_index, timestamp, frame_bgr) tuples.

        Parameters:
            source: Image, video path, webcam index (0), or RTSP URL ("rtsp://...").
            max_frames: Maximum number of frames to yield before stopping.
            start_time: Initial timestamp (defaults to datetime.now()).
        """
        curr_time = start_time or datetime.now()

        # Case 1: Single image input
        if isinstance(source, (np.ndarray, Image.Image)) or (
            isinstance(source, (str, Path)) and Path(str(source)).suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        ):
            frame = ImagePreprocessor.to_bgr_array(source)
            yield (0, curr_time, frame)
            return

        # Case 2: Video file, Webcam, or RTSP Stream via OpenCV VideoCapture
        src_repr = int(source) if isinstance(source, str) and source.isdigit() else source
        cap = cv2.VideoCapture(src_repr)

        if not cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or np.isnan(video_fps):
            video_fps = 25.0

        # Interval in source video frames between samples
        frame_step = max(1, int(round(video_fps / self.target_fps)))
        source_frame_idx = 0
        extracted_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if source_frame_idx % frame_step == 0:
                    time_offset_sec = source_frame_idx / video_fps
                    frame_ts = curr_time + timedelta(seconds=time_offset_sec)

                    yield (extracted_idx, frame_ts, frame)
                    extracted_idx += 1

                    if max_frames is not None and extracted_idx >= max_frames:
                        break

                source_frame_idx += 1
        finally:
            cap.release()
