"""
Land Waste & Object Detection Engine.
Supports Custom YOLO11 Trash Model, General Objects Model, and Combined Hybrid Mode.
"""
import time
from pathlib import Path
from typing import Union, Dict, Any, List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

try:
    from modelyolo.config import (
        MODEL_PATH,
        COCO_MODEL_PATH,
        DEFAULT_CONF_THRESHOLD,
        DEFAULT_IOU_THRESHOLD,
        DEFAULT_IMG_SIZE,
        TRASH_CLASS_NAMES,
        COMMON_OBJECT_MAPPING,
        CLASS_COLORS,
        get_waste_severity
    )
except ImportError:
    from config import (
        MODEL_PATH,
        COCO_MODEL_PATH,
        DEFAULT_CONF_THRESHOLD,
        DEFAULT_IOU_THRESHOLD,
        DEFAULT_IMG_SIZE,
        TRASH_CLASS_NAMES,
        COMMON_OBJECT_MAPPING,
        CLASS_COLORS,
        get_waste_severity
    )


class LandWasteDetector:
    """Detection engine for land waste, litter, and normal objects."""

    def __init__(
        self,
        trash_model_path: Union[str, Path] = MODEL_PATH,
        coco_model_path: Union[str, Path] = COCO_MODEL_PATH,
        conf: float = DEFAULT_CONF_THRESHOLD,
        iou: float = DEFAULT_IOU_THRESHOLD,
        detection_mode: str = "hybrid"  # "trash", "general", or "hybrid"
    ):
        self.conf = conf
        self.iou = iou
        self.detection_mode = detection_mode

        self.trash_model = None
        self.coco_model = None

        # Load custom trash model
        trash_path = Path(trash_model_path)
        if trash_path.exists():
            print(f"[*] Loading Custom Trash YOLO Model from: {trash_path.name}")
            self.trash_model = YOLO(str(trash_path))
            self.trash_raw_names = self.trash_model.names if hasattr(self.trash_model, 'names') else {}
        else:
            print(f"[!] Warning: Trash model not found at {trash_path}")

        # Load COCO general model for normal objects & everyday litter
        coco_path = Path(coco_model_path)
        if not coco_path.exists():
            coco_path = "yolo11n.pt"  # ultralytics will auto-download
        try:
            print(f"[*] Loading General Object YOLO Model: {coco_path}")
            self.coco_model = YOLO(str(coco_path))
            self.coco_raw_names = self.coco_model.names if hasattr(self.coco_model, 'names') else {}
        except Exception as e:
            print(f"[!] Warning: Could not load COCO model: {e}")

    def get_trash_class_name(self, cls_id: int) -> str:
        """Resolve custom trash class ID to label."""
        if cls_id in TRASH_CLASS_NAMES:
            return TRASH_CLASS_NAMES[cls_id]
        if hasattr(self, 'trash_raw_names') and cls_id in self.trash_raw_names:
            return self.trash_raw_names[cls_id]
        return f"Trash_Item_{cls_id}"

    def get_coco_class_name(self, cls_id: int) -> str:
        """Resolve COCO class name with friendly mapping."""
        if hasattr(self, 'coco_raw_names') and cls_id in self.coco_raw_names:
            raw_name = self.coco_raw_names[cls_id]
            if raw_name in COMMON_OBJECT_MAPPING:
                return COMMON_OBJECT_MAPPING[raw_name]
            return raw_name.capitalize()
        return f"Object_{cls_id}"

    def get_color_for_name(self, name: str, cls_id: int) -> Tuple[int, int, int]:
        """Generate consistent distinct color."""
        np.random.seed(abs(hash(name)) % (2**31 - 1))
        return tuple(int(c) for c in np.random.randint(60, 245, size=3))

    def detect(
        self,
        source: Union[str, Path, np.ndarray],
        conf: float = None,
        iou: float = None,
        mode: str = None,
        imgsz: int = DEFAULT_IMG_SIZE
    ) -> Dict[str, Any]:
        """
        Run detection on image or frame.
        mode can be 'trash', 'general', or 'hybrid' (both).
        """
        conf = conf if conf is not None else self.conf
        iou = iou if iou is not None else self.iou
        mode = mode if mode is not None else self.detection_mode

        start_time = time.perf_counter()
        detections: List[Dict[str, Any]] = []
        class_counts: Dict[str, int] = {}

        # 1. Run Custom Trash Model if active
        if (mode in ["trash", "hybrid"]) and self.trash_model is not None:
            results_trash = self.trash_model.predict(
                source=source,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False
            )
            boxes = results_trash[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy_list = boxes.xyxy.cpu().numpy()
                conf_list = boxes.conf.cpu().numpy()
                cls_list = boxes.cls.cpu().numpy().astype(int)

                for xyxy, score, cls_id in zip(xyxy_list, conf_list, cls_list):
                    name = self.get_trash_class_name(cls_id)
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    detections.append({
                        "class_id": int(cls_id),
                        "class_name": name,
                        "confidence": float(score),
                        "bbox": [x1, y1, x2, y2],
                        "source_model": "trash"
                    })
                    class_counts[name] = class_counts.get(name, 0) + 1

        # 2. Run General / COCO Model if active
        if (mode in ["general", "hybrid"]) and self.coco_model is not None:
            results_coco = self.coco_model.predict(
                source=source,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False
            )
            boxes = results_coco[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy_list = boxes.xyxy.cpu().numpy()
                conf_list = boxes.conf.cpu().numpy()
                cls_list = boxes.cls.cpu().numpy().astype(int)

                for xyxy, score, cls_id in zip(xyxy_list, conf_list, cls_list):
                    name = self.get_coco_class_name(cls_id)
                    x1, y1, x2, y2 = [int(v) for v in xyxy]

                    # Simple overlap suppression if already detected by trash model
                    is_duplicate = False
                    for existing in detections:
                        ex_x1, ex_y1, ex_x2, ex_y2 = existing["bbox"]
                        # Compute intersection over box
                        ix1, iy1 = max(x1, ex_x1), max(y1, ex_y1)
                        ix2, iy2 = min(x2, ex_x2), min(y2, ex_y2)
                        if ix2 > ix1 and iy2 > iy1:
                            inter_area = (ix2 - ix1) * (iy2 - iy1)
                            area = (x2 - x1) * (y2 - y1)
                            if inter_area / area > 0.6:
                                is_duplicate = True
                                break

                    if not is_duplicate:
                        detections.append({
                            "class_id": int(cls_id),
                            "class_name": name,
                            "confidence": float(score),
                            "bbox": [x1, y1, x2, y2],
                            "source_model": "general"
                        })
                        class_counts[name] = class_counts.get(name, 0) + 1

        inference_time_ms = (time.perf_counter() - start_time) * 1000
        total_items = len(detections)
        severity_label, severity_color = get_waste_severity(total_items)

        # Get base image array
        if isinstance(source, (str, Path)):
            orig_img = cv2.imread(str(source))
        else:
            orig_img = source.copy()

        annotated_image = self.draw_annotations(
            orig_img,
            detections,
            total_items,
            severity_label,
            severity_color,
            inference_time_ms
        )

        return {
            "total_items": total_items,
            "severity_label": severity_label,
            "severity_color": severity_color,
            "class_counts": class_counts,
            "detections": detections,
            "inference_time_ms": inference_time_ms,
            "annotated_image": annotated_image
        }

    def draw_annotations(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        total_items: int,
        severity_label: str,
        severity_color: Tuple[int, int, int],
        inference_time_ms: float
    ) -> np.ndarray:
        """Draw bounding boxes and heads-up banner."""
        annotated = image.copy()
        h, w, _ = annotated.shape

        for item in detections:
            x1, y1, x2, y2 = item["bbox"]
            name = item["class_name"]
            score = item["confidence"]
            cls_id = item["class_id"]
            color = self.get_color_for_name(name, cls_id)

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

            # Label text
            label = f"{name} {score:.0%}"
            font_scale = 0.55
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            label_y1 = max(0, y1 - text_h - baseline - 6)
            label_y2 = max(text_h + baseline + 6, y1)
            cv2.rectangle(
                annotated,
                (x1, label_y1),
                (x1 + text_w + 10, label_y2),
                color,
                -1
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 5, label_y2 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0) if sum(color) > 380 else (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA
            )

        # HUD Banner
        hud_h = 70
        hud_w = min(w, 420)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        cv2.circle(annotated, (28, 32), 7, severity_color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"STATUS: {severity_label}",
            (45, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            severity_color,
            2,
            lineType=cv2.LINE_AA
        )

        details_text = f"Objects: {total_items}  |  Speed: {inference_time_ms:.1f}ms"
        cv2.putText(
            annotated,
            details_text,
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            lineType=cv2.LINE_AA
        )

        return annotated
