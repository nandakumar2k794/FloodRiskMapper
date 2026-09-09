import logging
import base64
import time
import threading
import urllib.request
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

import cv2
import numpy as np
from pydantic import BaseModel
from fastapi import APIRouter, Query, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio

from src.db.mongo import get_db

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Singleton LandWasteDetector ────────────────────────────────────────────────
_detector = None


def get_detector():
    global _detector
    if _detector is None:
        try:
            import modelyolo.detector as my_detector
            _detector = my_detector.LandWasteDetector()
            logger.info("✅ DrainWatch LandWasteDetector initialized with YOLO11 model")
        except Exception as e:
            try:
                import modelyolo1.detector as my_detector
                _detector = my_detector.LandWasteDetector()
                logger.info("✅ DrainWatch LandWasteDetector initialized with YOLO11 model (modelyolo1)")
            except Exception as e2:
                logger.warning(f"⚠️ Could not load LandWasteDetector: {e2}")
    return _detector


class FrameDetectionRequest(BaseModel):
    image_base64: Optional[str] = None
    camera_id: str = "CAM_VELACHERY_01"
    conf_threshold: Optional[float] = 0.15


class PhoneStreamRequest(BaseModel):
    stream_url: str
    camera_id: str = "CAM_VELACHERY_01"
    conf_threshold: Optional[float] = 0.15


def _format_ip_url(url_or_ip: str) -> str:
    url = url_or_ip.strip()
    if not url.startswith(("http://", "https://", "rtsp://")):
        url = f"http://{url}"
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme or "http"
        netloc = parsed.netloc or parsed.path.split("/")[0]
        path = parsed.path if ":" in (parsed.netloc or "") else "/" + "/".join(parsed.path.split("/")[1:])

        host_part = netloc.split(":")[0] if ":" in netloc else netloc
        port_part = netloc.split(":")[1] if ":" in netloc else None

        # Auto-complete 3-octet IPs like 192.168.137 -> 192.168.137.1
        parts = host_part.split(".")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            host_part = f"{host_part}.1"

        if not port_part:
            port_part = "8080" if "8080" in url else "4747"

        netloc = f"{host_part}:{port_part}"
        if not path or path == "/":
            path = "/video"

        return f"{scheme}://{netloc}{path}"
    except Exception:
        return url


def _process_image_array(img_bgr: np.ndarray, camera_id: str, conf_threshold: Optional[float] = None) -> Dict[str, Any]:
    detector = get_detector()
    if detector is None:
        raise HTTPException(status_code=503, detail="YOLO detection engine unavailable")

    conf = conf_threshold if conf_threshold is not None else 0.15
    res = detector.detect(img_bgr, conf=conf)
    
    total_items = res.get("total_items", 0)
    class_counts = res.get("class_counts", {})
    detections = res.get("detections", [])
    annotated = res.get("annotated_image", img_bgr)
    severity_label = res.get("severity_label", "CLEAN LAND")
    
    # Calculate hydrological impact metrics
    # Higher count of trash/plastic/polythene directly maps to inlet blockage %
    blockage_pct = min(100.0, round(total_items * 14.5 + (5.0 if total_items > 0 else 0.0), 1))
    water_cov_pct = min(100.0, round(30.0 + blockage_pct * 0.5, 1))
    overflow_detected = blockage_pct >= 75.0 or (total_items >= 6)

    if overflow_detected:
        condition_class = "drain_overflowing"
        status_label = "CRITICAL OVERFLOW SURGE"
    elif blockage_pct >= 45.0:
        condition_class = "drain_blocked_garbage"
        status_label = "HEAVY DEBRIS CLOGGED"
    elif blockage_pct >= 15.0:
        condition_class = "standing_water_minor"
        status_label = "PARTIAL RESTRICTION"
    else:
        condition_class = "drain_clear"
        status_label = "CLEAR FLOW"

    # Encode annotated image to JPEG base64
    success, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    annotated_b64 = ""
    if success:
        annotated_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

    return {
        "camera_id": camera_id,
        "total_items": total_items,
        "class_counts": class_counts,
        "detections": detections,
        "severity_label": severity_label,
        "blockage_percentage": blockage_pct,
        "water_coverage_percentage": water_cov_pct,
        "overflow_detected": overflow_detected,
        "condition_class": condition_class,
        "status_label": status_label,
        "inference_time_ms": round(res.get("inference_time_ms", 0), 1),
        "annotated_image": annotated_b64,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/drainwatch/stations", summary="Get registered DrainWatch CCTV stations")
async def get_stations():
    """Returns all verified DrainWatch CCTV camera stations across Chennai."""
    try:
        from app.data_loader import get_chennai_cctv_cameras
        cctvs = get_chennai_cctv_cameras()
        return {
            "demo_station_id": "CAM_VELACHERY_01",
            "stations": cctvs,
            "total_stations": len(cctvs),
        }
    except Exception as e:
        logger.exception("Error loading CCTV stations")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drain-status", summary="Get latest DrainWatch observations")
async def get_drain_status(
    limit: int = Query(50, ge=1, le=200),
    condition: Optional[str] = Query(None, description="Filter by drain condition"),
    min_blockage: float = Query(0.0, ge=0.0, le=1.0),
):
    """Returns latest drain observations from registered cameras."""
    try:
        from app.data_loader import get_chennai_cctv_cameras
        observations = get_chennai_cctv_cameras()
        
        if condition:
            observations = [o for o in observations if o.get("condition_class") == condition]
        if min_blockage > 0:
            observations = [o for o in observations if (o.get("blockage_percentage", 0)/100.0) >= min_blockage]

        return {
            "observations": observations[:limit],
            "total_cameras": len(observations),
            "cameras_critical": sum(1 for o in observations if o.get("overflow_detected")),
            "cameras_blocked": sum(1 for o in observations if o.get("blockage_percentage", 0) >= 50),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.exception("Error fetching drain status")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/drainwatch/detect-frame", summary="Process a single frame through YOLO trash detection")
async def detect_frame(req: FrameDetectionRequest):
    """
    Runs YOLO11 trash & debris detection on a base64 frame (from browser webcam, phone stream, or snapshot).
    Returns real-time bounding boxes, item counts, blockage %, and annotated frame.
    """
    try:
        if not req.image_base64:
            raise HTTPException(status_code=400, detail="Empty frame: image_base64 is required.")

        cam_id = req.camera_id or "CAM_VELACHERY_01"
        b64_str = req.image_base64
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode image from base64.")

        return _process_image_array(img_bgr, cam_id, conf_threshold=req.conf_threshold)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Frame detection error")
        raise HTTPException(status_code=500, detail=str(e))


# ── Singleton Phone Stream Manager ─────────────────────────────────────────────
class PhoneStreamManager:
    """
    Ultra-reliable multi-threaded stream reader and real-time YOLO pipeline.
    Architecture:
    1. Dedicated Stream Reader Thread: Continuously reads raw frames from IP camera / DroidCam at 20-30 FPS.
       Never blocks on AI inference, ensuring 0ms latency and preventing TCP timeouts/socket drops.
    2. Dedicated Async AI Inference Thread: Asynchronously runs YOLO11 detection on the latest frame,
       updating bounding boxes and hydrological metrics without blocking video delivery.
    3. Real-Time Video Compositor: Overlays latest AI detections onto the fresh camera frames at 25 FPS.
    """
    def __init__(self):
        self._url = None
        self._running = False
        self._reader_thread = None
        self._infer_thread = None
        self._lock = threading.RLock()
        
        self._latest_raw_frame: Optional[np.ndarray] = None
        self._last_frame_time = 0.0
        self._frame_count = 0
        self._last_fps_calc = time.time()
        self._current_fps = 0.0
        self._conf_threshold = 0.15
        
        self._current_detections: List[Dict[str, Any]] = []
        self._current_total_items = 0
        self._current_severity_label = "CLEAN LAND"
        self._current_severity_color = (34, 197, 94)
        self._current_inference_ms = 0.0
        
        self._latest_telemetry: Dict[str, Any] = {
            "status": "IDLE",
            "camera_id": "CAM_VELACHERY_01",
            "total_items": 0,
            "class_counts": {},
            "blockage_percentage": 0.0,
            "water_coverage_percentage": 30.0,
            "overflow_detected": False,
            "status_label": "CLEAR FLOW",
            "inference_time_ms": 0.0,
            "fps": 0.0,
            "error": "",
        }

    def start(self, stream_url: str):
        formatted_url = _format_ip_url(stream_url)
        with self._lock:
            if self._running and self._url == formatted_url:
                return
        
        self.stop()
        
        with self._lock:
            self._url = formatted_url
            self._running = True
            # Reset frame state to prevent stale-frame freeze on reconnect
            self._latest_raw_frame = None
            self._last_frame_time = 0.0
            self._frame_count = 0
            self._last_fps_calc = time.time()
            self._latest_telemetry["status"] = "CONNECTING"
            self._latest_telemetry["error"] = ""
            
            self._reader_thread = threading.Thread(target=self._stream_reader_worker, daemon=True)
            self._infer_thread = threading.Thread(target=self._inference_worker, daemon=True)
            
            self._reader_thread.start()
            self._infer_thread.start()

    def stop(self):
        self._running = False
        for th in [self._reader_thread, self._infer_thread]:
            if th and th.is_alive() and threading.current_thread() != th:
                try:
                    th.join(timeout=0.6)
                except Exception:
                    pass
        with self._lock:
            self._latest_raw_frame = None
            self._latest_telemetry["status"] = "STOPPED"

    def get_latest_frame(self, timeout: float = 3.0) -> Optional[np.ndarray]:
        """Returns the freshest unannotated raw BGR frame."""
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                if self._latest_raw_frame is not None and (time.time() - self._last_frame_time < 4.0):
                    return self._latest_raw_frame.copy()
            time.sleep(0.04)
        return None

    def get_annotated_jpeg(self) -> Optional[bytes]:
        """Composites latest AI detection boxes onto the newest video frame and returns JPEG bytes."""
        with self._lock:
            if self._latest_raw_frame is None:
                return None
            if time.time() - self._last_frame_time > 4.5:
                return None
            frame = self._latest_raw_frame.copy()
            detections = list(self._current_detections)
            total_items = self._current_total_items
            sev_label = self._current_severity_label
            sev_color = self._current_severity_color
            inf_ms = self._current_inference_ms

        det = get_detector()
        if det is not None and detections:
            try:
                annotated = det.draw_annotations(
                    frame,
                    detections,
                    total_items,
                    sev_label,
                    sev_color,
                    inf_ms
                )
            except Exception:
                annotated = frame
        else:
            annotated = frame

        ret, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ret else None

    def get_telemetry(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest_telemetry)

    def set_telemetry(self, telem: Dict[str, Any]):
        with self._lock:
            self._latest_telemetry.update(telem)

    def set_conf_threshold(self, conf: float):
        with self._lock:
            self._conf_threshold = conf

    def _stream_reader_worker(self):
        url = self._url
        logger.info(f"[*] PhoneStreamManager reader started for: {url}")
        
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (4747 if ":4747" in url else 8080 if ":8080" in url else 4747)

        snapshot_candidates = [
            f"http://{host}:{port}/shot.jpg",
            f"http://{host}:{port}/cam/1/frame.jpg",
            f"http://{host}:{port}/photo.jpg",
        ]

        while self._running:
            # Try 1: VideoCapture stream (OpenCV)
            try:
                cap = cv2.VideoCapture(url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if cap.isOpened():
                    consecutive_failures = 0
                    while self._running:
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            consecutive_failures = 0
                            now = time.time()
                            with self._lock:
                                self._latest_raw_frame = frame
                                self._last_frame_time = now
                                self._frame_count += 1
                                self._latest_telemetry["status"] = "LIVE"
                                self._latest_telemetry["error"] = ""
                                elapsed = now - self._last_fps_calc
                                if elapsed >= 1.0:
                                    self._current_fps = round(self._frame_count / elapsed, 1)
                                    self._latest_telemetry["fps"] = self._current_fps
                                    self._frame_count = 0
                                    self._last_fps_calc = now
                        else:
                            consecutive_failures += 1
                            if consecutive_failures > 15:
                                break
                            time.sleep(0.04)
                    cap.release()
            except Exception as e:
                logger.debug(f"VideoCapture exception: {e}")

            # Try 2: Snapshot polling loop fallback
            if self._running:
                for snap_url in snapshot_candidates:
                    if not self._running:
                        break
                    success = self._poll_snapshots(snap_url)
                    if success:
                        break

            if self._running:
                with self._lock:
                    self._latest_telemetry["status"] = "RECONNECTING"
                time.sleep(1.0)

    def _poll_snapshots(self, snap_url: str) -> bool:
        """Polls direct snapshot endpoint at ~20 FPS."""
        req = urllib.request.Request(
            snap_url,
            headers={"User-Agent": "Mozilla/5.0 DrainWatch/1.0"}
        )
        consecutive_failures = 0
        while self._running and consecutive_failures < 8:
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=1.2) as resp:
                    if resp.status == 200:
                        arr = np.frombuffer(resp.read(), np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None and frame.size > 0:
                            consecutive_failures = 0
                            now = time.time()
                            with self._lock:
                                self._latest_raw_frame = frame
                                self._last_frame_time = now
                                self._frame_count += 1
                                self._latest_telemetry["status"] = "LIVE"
                                self._latest_telemetry["error"] = ""
                                elapsed = now - self._last_fps_calc
                                if elapsed >= 1.0:
                                    self._current_fps = round(self._frame_count / elapsed, 1)
                                    self._latest_telemetry["fps"] = self._current_fps
                                    self._frame_count = 0
                                    self._last_fps_calc = now
                        else:
                            consecutive_failures += 1
                    else:
                        consecutive_failures += 1
            except Exception:
                consecutive_failures += 1

            elapsed = time.time() - t0
            sleep_time = max(0.01, 0.045 - elapsed)
            time.sleep(sleep_time)

        return consecutive_failures < 8

    def _inference_worker(self):
        """Asynchronously processes raw frames with YOLO without lagging the video stream."""
        logger.info("[*] PhoneStreamManager AI inference worker active")
        while self._running:
            raw_frame = None
            with self._lock:
                if self._latest_raw_frame is not None and (time.time() - self._last_frame_time < 4.0):
                    raw_frame = self._latest_raw_frame.copy()

            if raw_frame is None:
                time.sleep(0.06)
                continue

            detector = get_detector()
            if detector is not None:
                try:
                    with self._lock:
                        current_conf = self._conf_threshold
                    t0 = time.perf_counter()
                    res = detector.detect(raw_frame, conf=current_conf, imgsz=416)
                    inf_ms = (time.perf_counter() - t0) * 1000

                    total_items = res.get("total_items", 0)
                    class_counts = res.get("class_counts", {})
                    detections = res.get("detections", [])
                    severity_label = res.get("severity_label", "CLEAN LAND")
                    severity_color = res.get("severity_color", (34, 197, 94))

                    blockage_pct = min(100.0, round(total_items * 14.5 + (5.0 if total_items > 0 else 0.0), 1))
                    water_cov_pct = min(100.0, round(30.0 + blockage_pct * 0.5, 1))
                    overflow_detected = blockage_pct >= 75.0 or (total_items >= 6)

                    if overflow_detected:
                        status_label = "CRITICAL OVERFLOW SURGE"
                    elif blockage_pct >= 45.0:
                        status_label = "HEAVY DEBRIS CLOGGED"
                    elif blockage_pct >= 15.0:
                        status_label = "PARTIAL RESTRICTION"
                    else:
                        status_label = "CLEAR FLOW"

                    with self._lock:
                        self._current_detections = detections
                        self._current_total_items = total_items
                        self._current_severity_label = severity_label
                        self._current_severity_color = severity_color
                        self._current_inference_ms = round(inf_ms, 1)

                        self._latest_telemetry.update({
                            "status": "LIVE",
                            "total_items": total_items,
                            "class_counts": class_counts,
                            "blockage_percentage": blockage_pct,
                            "water_coverage_percentage": water_cov_pct,
                            "overflow_detected": overflow_detected,
                            "status_label": status_label,
                            "inference_time_ms": round(inf_ms, 1),
                        })
                except Exception as e:
                    logger.debug(f"Live inference loop error: {e}")

            time.sleep(0.04)


_phone_stream_mgr = PhoneStreamManager()


def _generate_placeholder_frame(text: str = "CONNECTING TO PHONE CAMERA...") -> bytes:
    """Generates an initial dark placeholder frame while connecting."""
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (15, 23, 42)
    cv2.rectangle(img, (10, 10), (630, 350), (30, 41, 59), 2)
    cv2.putText(img, "DRAINWATCH AI - LIVE CAMERA FEED", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)
    cv2.putText(img, text, (40, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 1)
    ret, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ret else b""


def _fetch_direct_snapshot(raw_url: str) -> Optional[np.ndarray]:
    clean_url = raw_url.strip()
    candidates = []
    if ":8080" in clean_url:
        base = clean_url.split("/video")[0].rstrip("/")
        candidates.extend([f"{base}/shot.jpg", f"{base}/photo.jpg"])
    elif ":4747" in clean_url:
        base = clean_url.split("/video")[0].rstrip("/")
        candidates.append(f"{base}/cam/1/frame.jpg")
    elif any(clean_url.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
        candidates.append(clean_url)

    for cand in candidates:
        try:
            req = urllib.request.Request(
                cand,
                headers={"User-Agent": "Mozilla/5.0 DrainWatch/1.0"}
            )
            with urllib.request.urlopen(req, timeout=1.8) as resp:
                if resp.status == 200:
                    arr = np.frombuffer(resp.read(), np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None and img.size > 0:
                        return img
        except Exception:
            pass
    return None


@router.post("/drainwatch/upload-media", summary="Upload an image or frame for YOLO trash detection")
async def upload_media(
    file: UploadFile = File(...),
    camera_id: str = Query("CAM_VELACHERY_01"),
    conf_threshold: float = Query(0.15),
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode uploaded image.")
        return _process_image_array(img_bgr, camera_id, conf_threshold=conf_threshold)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload media detection error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/drainwatch/phone-stream", summary="Connect to Mobile IP Webcam stream and detect debris")
async def phone_stream(req: PhoneStreamRequest):
    url = _format_ip_url(req.stream_url)

    # 1. Fast path: Direct HTTP snapshot
    frame = _fetch_direct_snapshot(url)

    # 2. Stream path: Use PhoneStreamManager
    if frame is None:
        _phone_stream_mgr.start(url)
        frame = _phone_stream_mgr.get_latest_frame(timeout=3.0)

    # 3. Fallback: Direct VideoCapture single read
    if frame is None:
        try:
            cap = cv2.VideoCapture(url)
            ret, cv_frame = cap.read()
            cap.release()
            if ret and cv_frame is not None:
                frame = cv_frame
        except Exception:
            pass

    if frame is None:
        telem = _phone_stream_mgr.get_telemetry()
        err_msg = telem.get("error") or "Connection timed out."
        raise HTTPException(
            status_code=400,
            detail=f"Could not reach phone camera at {req.stream_url}. {err_msg} Ensure your phone and PC are connected to the same Wi-Fi and the camera app is broadcasting."
        )

    result = _process_image_array(frame, req.camera_id, conf_threshold=req.conf_threshold)
    _phone_stream_mgr.set_telemetry(result)
    return result


@router.post("/drainwatch/stop-stream", summary="Stop persistent phone camera stream")
async def stop_stream():
    _phone_stream_mgr.stop()
    return {"status": "STOPPED", "message": "Phone stream disconnected cleanly."}


@router.get("/drainwatch/phone-telemetry", summary="Get latest telemetry from active phone stream")
async def phone_telemetry(conf_threshold: Optional[float] = Query(None)):
    if conf_threshold is not None:
        _phone_stream_mgr.set_conf_threshold(conf_threshold)
    return _phone_stream_mgr.get_telemetry()


@router.get("/drainwatch/video-feed", summary="Live MJPEG video stream with real-time YOLO detection")
async def video_feed(stream_url: str = Query(...), camera_id: str = Query("CAM_VELACHERY_01")):
    """
    Continuous butter-smooth 25 FPS MJPEG video stream with real-time AI detections.
    Formats stream_url before starting the manager to prevent first-frame freeze.
    """
    formatted_url = _format_ip_url(stream_url)
    _phone_stream_mgr.start(formatted_url)

    async def generate_frames():
        while True:
            jpeg_bytes = _phone_stream_mgr.get_annotated_jpeg()
            
            if jpeg_bytes is None:
                telem = _phone_stream_mgr.get_telemetry()
                status = telem.get("status", "CONNECTING")
                if status == "RECONNECTING" or telem.get("error"):
                    p_bytes = _generate_placeholder_frame("RECONNECTING TO PHONE CAMERA...")
                else:
                    p_bytes = _generate_placeholder_frame("CONNECTING TO PHONE CAMERA...")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + p_bytes + b'\r\n')
                await asyncio.sleep(0.2)
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
            await asyncio.sleep(0.04)  # ~25 FPS smooth playback

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ── USB-C / External Camera Stream ─────────────────────────────────────────────
_usbc_lock = threading.RLock()
_usbc_state: Dict[str, Any] = {
    "running": False,
    "device_index": 0,
    "latest_frame": None,
    "last_frame_time": 0.0,
    "fps": 0.0,
    "frame_count": 0,
    "last_fps_calc": time.time(),
    "conf_threshold": 0.15,
    "detections": [],
    "total_items": 0,
    "severity_label": "CLEAN LAND",
    "severity_color": (34, 197, 94),
    "inference_ms": 0.0,
    "telemetry": {
        "status": "IDLE",
        "total_items": 0,
        "class_counts": {},
        "blockage_percentage": 0.0,
        "water_coverage_percentage": 30.0,
        "overflow_detected": False,
        "status_label": "CLEAR FLOW",
        "inference_time_ms": 0.0,
        "fps": 0.0,
    },
}
_usbc_reader_thread: Optional[threading.Thread] = None
_usbc_infer_thread: Optional[threading.Thread] = None


def _usbc_reader_worker(device_index: int):
    """Dedicated reader thread for USB-C / external camera."""
    global _usbc_state
    logger.info(f"[*] USB-C reader started for device {device_index}")
    while True:
        with _usbc_lock:
            if not _usbc_state["running"]:
                break
        try:
            cap = cv2.VideoCapture(device_index)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            if not cap.isOpened():
                with _usbc_lock:
                    _usbc_state["telemetry"]["status"] = "DEVICE_NOT_FOUND"
                time.sleep(2.0)
                cap.release()
                continue
            consecutive_failures = 0
            while True:
                with _usbc_lock:
                    if not _usbc_state["running"]:
                        break
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    consecutive_failures = 0
                    now = time.time()
                    with _usbc_lock:
                        _usbc_state["latest_frame"] = frame
                        _usbc_state["last_frame_time"] = now
                        _usbc_state["frame_count"] += 1
                        _usbc_state["telemetry"]["status"] = "LIVE"
                        elapsed = now - _usbc_state["last_fps_calc"]
                        if elapsed >= 1.0:
                            fps = round(_usbc_state["frame_count"] / elapsed, 1)
                            _usbc_state["fps"] = fps
                            _usbc_state["telemetry"]["fps"] = fps
                            _usbc_state["frame_count"] = 0
                            _usbc_state["last_fps_calc"] = now
                else:
                    consecutive_failures += 1
                    if consecutive_failures > 20:
                        break
                    time.sleep(0.03)
            cap.release()
        except Exception as e:
            logger.debug(f"USB-C reader error: {e}")
            time.sleep(1.0)
    logger.info("[*] USB-C reader stopped.")


def _usbc_inference_worker():
    """Dedicated AI inference thread for USB-C camera frames."""
    global _usbc_state
    logger.info("[*] USB-C inference worker active")
    while True:
        with _usbc_lock:
            if not _usbc_state["running"]:
                break
            raw_frame = None
            if _usbc_state["latest_frame"] is not None and (time.time() - _usbc_state["last_frame_time"] < 5.0):
                raw_frame = _usbc_state["latest_frame"].copy()
        if raw_frame is None:
            time.sleep(0.06)
            continue
        detector = get_detector()
        if detector is not None:
            try:
                with _usbc_lock:
                    current_conf = _usbc_state["conf_threshold"]
                t0 = time.perf_counter()
                res = detector.detect(raw_frame, conf=current_conf, imgsz=416)
                inf_ms = (time.perf_counter() - t0) * 1000
                total_items = res.get("total_items", 0)
                class_counts = res.get("class_counts", {})
                detections = res.get("detections", [])
                severity_label = res.get("severity_label", "CLEAN LAND")
                severity_color = res.get("severity_color", (34, 197, 94))
                blockage_pct = min(100.0, round(total_items * 14.5 + (5.0 if total_items > 0 else 0.0), 1))
                water_cov_pct = min(100.0, round(30.0 + blockage_pct * 0.5, 1))
                overflow_detected = blockage_pct >= 75.0 or total_items >= 6
                status_label = "CRITICAL OVERFLOW SURGE" if overflow_detected else "HEAVY DEBRIS CLOGGED" if blockage_pct >= 45.0 else "PARTIAL RESTRICTION" if blockage_pct >= 15.0 else "CLEAR FLOW"
                with _usbc_lock:
                    _usbc_state["detections"] = detections
                    _usbc_state["total_items"] = total_items
                    _usbc_state["severity_label"] = severity_label
                    _usbc_state["severity_color"] = severity_color
                    _usbc_state["inference_ms"] = round(inf_ms, 1)
                    _usbc_state["telemetry"].update({
                        "status": "LIVE",
                        "total_items": total_items,
                        "class_counts": class_counts,
                        "blockage_percentage": blockage_pct,
                        "water_coverage_percentage": water_cov_pct,
                        "overflow_detected": overflow_detected,
                        "status_label": status_label,
                        "inference_time_ms": round(inf_ms, 1),
                    })
            except Exception as e:
                logger.debug(f"USB-C inference error: {e}")
        time.sleep(0.04)
    logger.info("[*] USB-C inference worker stopped.")


def _start_usbc(device_index: int):
    global _usbc_reader_thread, _usbc_infer_thread, _usbc_state
    with _usbc_lock:
        if _usbc_state["running"] and _usbc_state["device_index"] == device_index:
            return
    _stop_usbc()
    with _usbc_lock:
        _usbc_state["running"] = True
        _usbc_state["device_index"] = device_index
        _usbc_state["latest_frame"] = None
        _usbc_state["last_frame_time"] = 0.0
        _usbc_state["frame_count"] = 0
        _usbc_state["last_fps_calc"] = time.time()
        _usbc_state["telemetry"]["status"] = "CONNECTING"
    _usbc_reader_thread = threading.Thread(target=_usbc_reader_worker, args=(device_index,), daemon=True)
    _usbc_infer_thread = threading.Thread(target=_usbc_inference_worker, daemon=True)
    _usbc_reader_thread.start()
    _usbc_infer_thread.start()


def _stop_usbc():
    global _usbc_reader_thread, _usbc_infer_thread, _usbc_state
    with _usbc_lock:
        _usbc_state["running"] = False
        _usbc_state["telemetry"]["status"] = "STOPPED"
    for th in [_usbc_reader_thread, _usbc_infer_thread]:
        if th and th.is_alive():
            try:
                th.join(timeout=1.0)
            except Exception:
                pass


def _get_usbc_jpeg() -> Optional[bytes]:
    with _usbc_lock:
        frame = _usbc_state.get("latest_frame")
        if frame is None or (time.time() - _usbc_state["last_frame_time"]) > 5.0:
            return None
        frame = frame.copy()
        detections = list(_usbc_state.get("detections", []))
        total_items = _usbc_state.get("total_items", 0)
        sev_label = _usbc_state.get("severity_label", "CLEAN LAND")
        sev_color = _usbc_state.get("severity_color", (34, 197, 94))
        inf_ms = _usbc_state.get("inference_ms", 0.0)
    det = get_detector()
    if det is not None and detections:
        try:
            annotated = det.draw_annotations(frame, detections, total_items, sev_label, sev_color, inf_ms)
        except Exception:
            annotated = frame
    else:
        annotated = frame
    ret, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ret else None


@router.get("/drainwatch/usbc-feed", summary="Live MJPEG feed from USB-C external camera with YOLO detection")
async def usbc_feed(device_index: int = Query(0, ge=0, le=10)):
    """
    Streams live MJPEG video from a USB-C connected external camera (OpenCV device index).
    Decoupled reader + inference threads prevent freezing. Supports hot-plug USB-C cameras.
    """
    _start_usbc(device_index)

    async def generate_frames():
        while True:
            jpeg_bytes = _get_usbc_jpeg()
            if jpeg_bytes is None:
                with _usbc_lock:
                    status = _usbc_state["telemetry"].get("status", "CONNECTING")
                if status == "DEVICE_NOT_FOUND":
                    p_bytes = _generate_placeholder_frame(f"USB-C DEVICE {device_index} NOT FOUND")
                else:
                    p_bytes = _generate_placeholder_frame(f"CONNECTING USB-C CAMERA (INDEX {device_index})...")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + p_bytes + b'\r\n')
                await asyncio.sleep(0.25)
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
            await asyncio.sleep(0.033)  # ~30 FPS

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.post("/drainwatch/usbc-stop", summary="Stop USB-C camera stream")
async def usbc_stop():
    _stop_usbc()
    return {"status": "STOPPED", "message": "USB-C camera stream stopped cleanly."}


@router.get("/drainwatch/usbc-telemetry", summary="Get live YOLO telemetry from USB-C camera")
async def usbc_telemetry(conf_threshold: Optional[float] = Query(None)):
    with _usbc_lock:
        if conf_threshold is not None:
            _usbc_state["conf_threshold"] = conf_threshold
        return dict(_usbc_state["telemetry"])


@router.get("/drainwatch/discover-phone", summary="Scan local network and USB interfaces for active phone cameras")
async def discover_phone():
    """
    Rapidly probes candidate IPs (USB loopback, hotspot, local Wi-Fi subnets) on ports 4747 (DroidCam) and 8080 (IP Webcam).
    Returns list of reachable streams and suggested endpoints.
    """
    import socket
    from concurrent.futures import ThreadPoolExecutor

    candidates = [
        {"type": "USB / ADB Loopback (DroidCam)", "url": "http://127.0.0.1:4747/video", "ip": "127.0.0.1", "port": 4747},
        {"type": "USB / ADB Loopback (IP Webcam)", "url": "http://127.0.0.1:8080/video", "ip": "127.0.0.1", "port": 8080},
    ]

    subnets_to_probe = {"192.168.137", "192.168.1", "192.168.43", "192.168.0", "192.168.56"}
    try:
        hostname = socket.gethostname()
        local_ips = socket.gethostbyname_ex(hostname)[2]
        for lip in local_ips:
            if not lip.startswith(("127.", "169.254.")):
                sub = ".".join(lip.split(".")[:3])
                subnets_to_probe.add(sub)
    except Exception:
        pass

    for sub in subnets_to_probe:
        # Common gateway / device IPs
        for i in range(1, 255):
            ip = f"{sub}.{i}"
            candidates.append({"type": f"Phone ({ip}:8080)", "url": f"http://{ip}:8080/video", "ip": ip, "port": 8080})
            candidates.append({"type": f"Phone ({ip}:4747)", "url": f"http://{ip}:4747/video", "ip": ip, "port": 4747})

    def check_candidate(c):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.12)
        try:
            res = sock.connect_ex((c["ip"], c["port"]))
            if res == 0:
                return {**c, "reachable": True}
        except Exception:
            pass
        finally:
            sock.close()
        return None

    with ThreadPoolExecutor(max_workers=60) as executor:
        results = list(executor.map(check_candidate, candidates))

    reachable_items = [r for r in results if r is not None]
    return {
        "found": len(reachable_items) > 0,
        "active_cameras": reachable_items,
        "suggested_ips": [f"http://192.168.137.1:8080/video", "http://192.168.137.1:4747/video", "http://127.0.0.1:4747/video"],
    }





