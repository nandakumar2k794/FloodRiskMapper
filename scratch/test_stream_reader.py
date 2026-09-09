import urllib.request
import numpy as np
import cv2
import time
import threading

class RobustStreamReader:
    """
    Robust reader for Mobile IP Webcam and DroidCam streams.
    Supports:
    1. HTTP MJPEG stream (e.g. http://ip:8080/video or http://ip:4747/video) via direct HTTP stream.
    2. Snapshot polling (e.g. http://ip:8080/shot.jpg or http://ip:4747/cam/1/frame.jpg).
    3. OpenCV fallback for RTSP / webcam devices.
    """
    def __init__(self):
        self._url = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._last_frame_time = 0
        self._status = "IDLE"
        self._error = ""

    def start(self, url: str):
        with self._lock:
            if self._running and self._url == url:
                return
            self.stop()
            self._url = url.strip()
            self._running = True
            self._status = "CONNECTING"
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        with self._lock:
            self._latest_frame = None
            self._status = "STOPPED"

    def get_frame(self, max_age=3.0):
        with self._lock:
            if self._latest_frame is not None and (time.time() - self._last_frame_time <= max_age):
                return self._latest_frame.copy(), self._status, None
            return None, self._status, self._error

    def _run(self):
        url = self._url
        print(f"[*] RobustStreamReader connecting to: {url}")
        
        # Determine mode:
        is_snapshot_url = any(url.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]) or "shot.jpg" in url or "frame.jpg" in url
        is_rtsp = url.startswith("rtsp://")

        while self._running:
            if is_snapshot_url:
                self._poll_snapshots(url)
            elif is_rtsp:
                self._stream_opencv(url)
            else:
                # Try direct HTTP MJPEG first (most stable for Android IP Webcam and DroidCam)
                success = self._stream_mjpeg(url)
                if not success and self._running:
                    print(f"[*] MJPEG direct stream failed, trying OpenCV fallback for: {url}")
                    self._stream_opencv(url)
            
            if self._running:
                with self._lock:
                    self._status = "RECONNECTING"
                time.sleep(1.0)

    def _stream_mjpeg(self, url: str) -> bool:
        """Reads HTTP MJPEG stream directly without OpenCV ffmpeg buffer lag."""
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DrainWatch/1.0",
                    "Accept": "multipart/x-mixed-replace, image/jpeg, */*",
                }
            )
            with urllib.request.urlopen(req, timeout=5.0) as response:
                with self._lock:
                    self._status = "CONNECTED"
                    self._error = ""

                bytes_buffer = bytearray()
                while self._running:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    bytes_buffer.extend(chunk)

                    # Look for complete JPEG between SOI (0xFF, 0xD8) and EOI (0xFF, 0xD9)
                    a = bytes_buffer.find(b'\xff\xd8')
                    b = bytes_buffer.find(b'\xff\xd9')
                    if a != -1 and b != -1:
                        if b > a:
                            jpg_bytes = bytes_buffer[a:b+2]
                            bytes_buffer = bytes_buffer[b+2:]
                            
                            # Fast decode
                            arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if frame is not None:
                                with self._lock:
                                    self._latest_frame = frame
                                    self._last_frame_time = time.time()
                                    self._status = "LIVE"
                        else:
                            # EOI came before SOI, discard broken prefix
                            bytes_buffer = bytes_buffer[b+2:]

                    # Keep buffer from growing unbounded if stream drops
                    if len(bytes_buffer) > 200000:
                        bytes_buffer = bytearray()

                return True
        except Exception as e:
            with self._lock:
                self._error = str(e)
            print(f"[!] MJPEG stream error: {e}")
            return False

    def _stream_opencv(self, url: str):
        """OpenCV fallback reader."""
        cap = None
        try:
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                with self._lock:
                    self._status = "ERROR"
                    self._error = "OpenCV could not open stream URL"
                return

            with self._lock:
                self._status = "CONNECTED"

            while self._running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

                with self._lock:
                    self._latest_frame = frame
                    self._last_frame_time = time.time()
                    self._status = "LIVE"
                time.sleep(0.02)
        except Exception as e:
            with self._lock:
                self._error = str(e)
        finally:
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass

    def _poll_snapshots(self, url: str):
        """Polls snapshot endpoint periodically."""
        while self._running:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "DrainWatch/1.0"})
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    if resp.status == 200:
                        arr = np.frombuffer(resp.read(), np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self._lock:
                                self._latest_frame = frame
                                self._last_frame_time = time.time()
                                self._status = "LIVE"
            except Exception as e:
                with self._lock:
                    self._error = str(e)
            time.sleep(0.1)

print("RobustStreamReader defined successfully")
