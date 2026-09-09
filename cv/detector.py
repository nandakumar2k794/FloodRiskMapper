"""
YOLO Object Detection Interface for DrainWatch AI.
Supports lightweight pretrained YOLOv8/v11 models and custom weights hot-swapping.
Detects 13 target classes for drain blockages, garbage accumulation, and waterlogging.

Important:
- Water depth in cm is NEVER claimed without calibrated camera geometry.
- overflow_detected uses a multi-evidence heuristic: direct class detection takes
  precedence over percentage thresholds to minimise false-positive overrides.
- road_waterlogged is the highest severity condition (score=1.0).
"""
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2

from cv.models import (
    DrainWatchClass,
    DetectionBox,
    FrameObservation,
    BLOCKAGE_CLASSES,
    DRAIN_STRUCTURE_CLASSES,
    WATER_CLASSES,
)
from cv.preprocessing import ImagePreprocessor, ROISelector
from config.constants import DRAINWATCH_CLASS_SCORES


# Class mapping dictionary from common COCO / general object classes to DrainWatch categories
COCO_PROXY_CLASS_MAP = {
    "bottle": DrainWatchClass.PLASTIC_WASTE.value,
    "cup": DrainWatchClass.PLASTIC_WASTE.value,
    "handbag": DrainWatchClass.GARBAGE.value,
    "backpack": DrainWatchClass.GARBAGE.value,
    "suitcase": DrainWatchClass.GARBAGE.value,
    "bowl": DrainWatchClass.GARBAGE.value,
    "umbrella": DrainWatchClass.GARBAGE.value,
    "potted plant": DrainWatchClass.LEAVES.value,
    "bench": DrainWatchClass.COVERED_DRAIN.value,
}


class DrainDetector:
    """
    YOLO Inference and visual drain condition analyzer.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        use_clahe: bool = True,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.use_clahe = use_clahe
        self.model = None

        self._init_yolo_model()

    def _init_yolo_model(self) -> None:
        """Initialize YOLO model instance."""
        try:
            from ultralytics import YOLO
            if self.weights_path and self.weights_path.exists():
                self.model = YOLO(str(self.weights_path))
            else:
                # Use standard lightweight pretrained nano model by default
                self.model = YOLO("yolov8n.pt")
        except Exception:
            # Fallback for environments without ultralytics model download
            self.model = None

    def load_weights(self, weights_path: Union[str, Path]) -> None:
        """
        Hot-swap model weights with custom trained domain weights.
        """
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"Custom YOLO weights file not found: {path}")

        from ultralytics import YOLO
        self.weights_path = path
        self.model = YOLO(str(path))

    def detect_frame(
        self,
        frame_input: Union[str, Path, np.ndarray],
        roi: Optional[ROISelector] = None,
    ) -> List[DetectionBox]:
        """
        Run object detection on an image frame.

        Parameters:
            frame_input: Image array or path.
            roi: Optional ROISelector instance.

        Returns:
            List[DetectionBox]: Detected bounding boxes.
        """
        frame_bgr = ImagePreprocessor.to_bgr_array(frame_input)

        if self.use_clahe:
            frame_bgr = ImagePreprocessor.enhance_contrast(frame_bgr)

        detections: List[DetectionBox] = []

        if self.model is None:
            return detections

        try:
            results = self.model.predict(
                source=frame_bgr,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )

            if results and len(results) > 0:
                res = results[0]
                names = res.names
                boxes = res.boxes

                for box in boxes:
                    cls_id = int(box.cls[0])
                    raw_cls_name = names.get(cls_id, str(cls_id)).lower()
                    conf = float(box.conf[0])
                    xyxy = [float(coord) for coord in box.xyxy[0]]
                    x1, y1, x2, y2 = xyxy

                    # Map standard proxy classes if needed
                    target_cls = COCO_PROXY_CLASS_MAP.get(raw_cls_name, raw_cls_name)

                    area = max(0.0, (x2 - x1) * (y2 - y1))

                    # Check ROI filtering
                    in_roi = True
                    if roi is not None:
                        in_roi = roi.is_box_in_roi((x1, y1, x2, y2), frame_bgr.shape)

                    detections.append(
                        DetectionBox(
                            class_name=target_cls,
                            confidence=round(conf, 3),
                            box=(x1, y1, x2, y2),
                            area=round(area, 1),
                            is_in_roi=in_roi,
                            metadata={"raw_class": raw_cls_name},
                        )
                    )
        except Exception:
            pass

        return detections

    def process_frame(
        self,
        frame_input: Union[str, Path, np.ndarray],
        camera_id: str = "camera_01",
        timestamp: Optional[datetime] = None,
        frame_index: int = 0,
        roi: Optional[ROISelector] = None,
    ) -> FrameObservation:
        """
        End-to-end frame processing generating a structured FrameObservation.
        Computes blockage percentage, water coverage, overflow status, and condition score.
        """
        frame_bgr = ImagePreprocessor.to_bgr_array(frame_input)
        h, w = frame_bgr.shape[:2]
        total_frame_area = float(h * w)

        detections = self.detect_frame(frame_bgr, roi=roi)
        ts = timestamp or datetime.now()

        # Filter detections inside ROI (or all if no ROI)
        active_detections = [d for d in detections if d.is_in_roi]

        # Calculate areas
        garbage_boxes = [d for d in active_detections if d.class_name in BLOCKAGE_CLASSES]
        drain_boxes = [d for d in active_detections if d.class_name in DRAIN_STRUCTURE_CLASSES]
        water_boxes = [d for d in active_detections if d.class_name in WATER_CLASSES]

        # Total intersection area with ROI
        roi_selector = roi or ROISelector()
        garbage_area = sum(roi_selector.compute_overlap_area(d.box, frame_bgr.shape) for d in garbage_boxes)
        drain_area = sum(roi_selector.compute_overlap_area(d.box, frame_bgr.shape) for d in drain_boxes)
        water_area = sum(roi_selector.compute_overlap_area(d.box, frame_bgr.shape) for d in water_boxes)

        # Baseline inspection area: ROI area or detected drain area or frame fraction
        abs_roi = roi_selector.get_absolute_box(frame_bgr.shape)
        if abs_roi:
            inspection_area = max(1.0, (abs_roi[2] - abs_roi[0]) * (abs_roi[3] - abs_roi[1]))
        elif drain_area > 0:
            inspection_area = drain_area
        else:
            inspection_area = total_frame_area * 0.25  # default baseline region

        # Blockage percentage: (garbage area / inspection area) * 100
        blockage_pct = float(np.clip((garbage_area / inspection_area) * 100.0, 0.0, 100.0))

        # Water coverage percentage
        water_pct = float(np.clip((water_area / inspection_area) * 100.0, 0.0, 100.0))

        # Flags — use direct detection classes first, then percentage thresholds as secondary evidence
        has_garbage = len(garbage_boxes) > 0 or blockage_pct > 5.0

        # Overflow: prioritise direct class detection; percentage threshold is a weaker secondary signal
        direct_overflow_classes = {
            DrainWatchClass.OVERFLOWING_DRAIN.value,
            DrainWatchClass.ROAD_WATERLOGGING.value,
        }
        has_direct_overflow = any(
            d.class_name in direct_overflow_classes for d in active_detections
        )
        # Secondary heuristic: very high blockage AND significant water simultaneously
        heuristic_overflow = blockage_pct > 70.0 and water_pct > 50.0
        has_overflow = has_direct_overflow or heuristic_overflow

        # Calculate mean detection confidence
        confs = [d.confidence for d in active_detections]
        avg_conf = float(np.mean(confs)) if confs else 0.85

        # Determine condition class & continuous score (priority ordered, highest first)
        # road_waterlogged is the highest-severity class (score=1.0)
        has_road_waterlogging = any(
            d.class_name == DrainWatchClass.ROAD_WATERLOGGING.value for d in active_detections
        )
        if has_road_waterlogging or (has_overflow and water_pct >= 60.0):
            condition_class = "road_waterlogged"
            condition_score = DRAINWATCH_CLASS_SCORES["road_waterlogged"]
        elif has_overflow:
            condition_class = "drain_overflowing"
            condition_score = DRAINWATCH_CLASS_SCORES["drain_overflowing"]
        elif blockage_pct >= 40.0:
            condition_class = "drain_blocked_garbage"
            condition_score = DRAINWATCH_CLASS_SCORES["drain_blocked_garbage"]
        elif blockage_pct >= 20.0 or has_garbage:
            condition_class = "drain_blocked_silt"
            condition_score = DRAINWATCH_CLASS_SCORES["drain_blocked_silt"]
        elif water_pct >= 50.0:
            condition_class = "standing_water_major"
            condition_score = DRAINWATCH_CLASS_SCORES["standing_water_major"]
        elif water_pct >= 15.0:
            condition_class = "standing_water_minor"
            condition_score = DRAINWATCH_CLASS_SCORES["standing_water_minor"]
        else:
            condition_class = "drain_clear"
            condition_score = DRAINWATCH_CLASS_SCORES["drain_clear"]

        # Normalized areas: fraction of inspection zone occupied [0.0, 1.0]
        normalized_blockage = float(np.clip(garbage_area / inspection_area, 0.0, 1.0))
        normalized_water = float(np.clip(water_area / inspection_area, 0.0, 1.0))

        return FrameObservation(
            camera_id=camera_id,
            timestamp=ts,
            frame_index=frame_index,
            garbage_detected=has_garbage,
            garbage_area=round(garbage_area, 1),
            drain_area=round(drain_area, 1),
            blockage_percentage=round(blockage_pct, 1),
            water_coverage_percentage=round(water_pct, 1),
            overflow_detected=has_overflow,
            confidence=round(avg_conf, 3),
            condition_class=condition_class,
            condition_score=round(condition_score, 2),
            detections=detections,
            water_depth_cm=None,  # NEVER claimed without calibrated geometry/markers
            calibration_required=True,
            normalized_blockage_area=round(normalized_blockage, 4),
            normalized_water_area=round(normalized_water, 4),
        )
