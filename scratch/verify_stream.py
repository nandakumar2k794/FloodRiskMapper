import time
import threading
import numpy as np
import cv2
from http.server import HTTPServer, BaseHTTPRequestHandler
from scratch.test_stream_reader import RobustStreamReader

class MockMJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/video":
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            
            frame_idx = 0
            while True:
                # Generate a dummy frame with counter
                img = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(img, f"Frame {frame_idx}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                ret, jpeg = cv2.imencode('.jpg', img)
                if not ret:
                    break
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
                frame_idx += 1
                time.sleep(0.033) # 30 fps
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass # quiet

server = HTTPServer(('127.0.0.1', 8999), MockMJPEGHandler)
server_thread = threading.Thread(target=server.serve_forever, daemon=True)
server_thread.start()
print("[*] Mock MJPEG server running on http://127.0.0.1:8999/video")

reader = RobustStreamReader()
reader.start("http://127.0.0.1:8999/video")

time.sleep(0.5)

frames_read = 0
for i in range(10):
    frame, status, err = reader.get_frame()
    if frame is not None:
        frames_read += 1
    time.sleep(0.05)

print(f"[*] Frames successfully read: {frames_read}/10, Status: {status}")
reader.stop()
server.shutdown()
print("[*] Test completed successfully!")
