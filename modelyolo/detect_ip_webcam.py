"""
Real-time Mobile IP Webcam Stream Detection.
Connects to an IP Webcam app (Android/iOS) over Wi-Fi/Hotspot
and performs real-time YOLO11 waste & object detection on your laptop screen.

Common mobile apps supported:
- IP Webcam (Android) -> http://<phone_ip>:8080/video
- DroidCam -> http://<phone_ip>:4747/video
- Iriun Webcam
- RTSP / HTTP streams
"""
import argparse
import time
from pathlib import Path
from urllib.parse import urlparse
import cv2

from detector import LandWasteDetector
from config import DEFAULT_CONF_THRESHOLD, OUTPUT_DIR


def format_ip_url(url_or_ip: str) -> str:
    """Format and normalize user input into a valid stream URL."""
    url = url_or_ip.strip()
    if not url.startswith(("http://", "https://", "rtsp://")):
        url = f"http://{url}"

    parsed = urlparse(url)
    # If no path specified on standard IP Webcam port 8080 or 4747, append /video
    if not parsed.path or parsed.path == "/":
        if parsed.port == 4747:
            url = f"{url.rstrip('/')}/video"
        else:
            url = f"{url.rstrip('/')}/video"

    return url


def run_ip_webcam_detection(
    stream_url: str,
    conf: float = DEFAULT_CONF_THRESHOLD,
    save_output: bool = False,
    window_name: str = "Mobile IP Camera - Land Waste Detection (YOLO11)"
):
    target_url = format_ip_url(stream_url)
    print(f"\n[*] Connecting to Mobile IP Camera stream: {target_url}")
    print("[*] Make sure your phone and laptop are on the same Wi-Fi / Mobile Hotspot network.")
    print("[*] Controls:")
    print("    - [q] or [ESC] : Exit")
    print("    - [+] / [-]    : Increase / Decrease Confidence threshold")
    print("    - [s]          : Save snapshot of current detection")
    print("    - [h]          : Toggle HUD overlay")
    print("    - [f]          : Toggle Fullscreen")

    detector = LandWasteDetector(conf=conf)

    # Use OpenCV VideoCapture with buffer size 1 for low latency
    cap = cv2.VideoCapture(target_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"\n[!] ERROR: Could not open stream from '{target_url}'.")
        print("[!] Troubleshooting tips:")
        print("    1. Verify the IP address and port displayed on your mobile IP Webcam screen.")
        print("    2. Ensure both laptop and phone are connected to the SAME Wi-Fi or phone hotspot.")
        print("    3. If your phone shows 'http://192.168.1.X:8080', try running:")
        print(f"       python detect_ip_webcam.py --url http://192.168.1.X:8080/video")
        return

    print("[+] Connected to Mobile Camera successfully!")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)

    video_writer = None
    if save_output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_file = OUTPUT_DIR / f"ip_webcam_record_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(out_file), fourcc, 20.0, (width, height))
        print(f"[*] Recording detection video to: {out_file}")

    prev_time = time.time()
    fps_smooth = 0.0
    current_conf = conf
    show_hud = True
    is_fullscreen = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Lost connection or no frame received. Retrying...")
                time.sleep(0.5)
                continue

            # Calculate FPS
            curr_time = time.time()
            elapsed = curr_time - prev_time
            prev_time = curr_time
            if elapsed > 0:
                fps_smooth = 0.85 * fps_smooth + 0.15 * (1.0 / elapsed)

            # Perform YOLO Land Waste Detection
            results = detector.detect(frame, conf=current_conf)
            annotated_frame = results["annotated_image"] if show_hud else frame

            # Top bar live stats
            status_text = f"Live Mobile Stream | Conf: {current_conf:.2f} | FPS: {fps_smooth:.1f}"
            cv2.putText(
                annotated_frame,
                status_text,
                (annotated_frame.shape[1] - 440, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Record if enabled
            if video_writer:
                video_writer.write(annotated_frame)

            # Display on laptop screen
            cv2.imshow(window_name, annotated_frame)

            # Key controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Exit
                print("[*] Stream stopped by user.")
                break
            elif key == ord('+') or key == ord('='):  # Increase confidence
                current_conf = min(0.95, current_conf + 0.05)
                print(f"[*] Confidence threshold: {current_conf:.2f}")
            elif key == ord('-') or key == ord('_'):  # Decrease confidence
                current_conf = max(0.05, current_conf - 0.05)
                print(f"[*] Confidence threshold: {current_conf:.2f}")
            elif key == ord('h'):  # Toggle HUD
                show_hud = not show_hud
            elif key == ord('f'):  # Toggle Fullscreen
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            elif key == ord('s'):  # Save Snapshot
                snap_time = time.strftime("%Y%m%d_%H%M%S")
                snap_path = OUTPUT_DIR / f"snapshot_{snap_time}.jpg"
                cv2.imwrite(str(snap_path), annotated_frame)
                print(f"[+] Snapshot saved: {snap_path}")

    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Stream phone camera via IP Webcam and detect waste on land.")
    parser.add_argument("--url", type=str, default=None,
                        help="IP Webcam URL (e.g., http://192.168.1.5:8080/video or 192.168.1.5:8080)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD,
                        help=f"Confidence threshold (default: {DEFAULT_CONF_THRESHOLD})")
    parser.add_argument("--save", action="store_true",
                        help="Save recorded detection video to runs/detect/")

    args = parser.parse_args()

    url = args.url
    if not url:
        print("=" * 65)
        print("     CONNECT MOBILE IP WEBCAM TO LAPTOP (YOLO11)")
        print("=" * 65)
        print("1. Open the 'IP Webcam' app on your phone.")
        print("2. Scroll down and tap 'Start server'.")
        print("3. Look at the IP address shown on the phone screen.")
        print("   (Example: http://192.168.1.10:8080 or 192.168.43.1:8080)\n")
        url = input("Enter your phone's IP Webcam URL/Address: ").strip()

    if not url:
        print("[!] No URL provided. Exiting.")
        return

    run_ip_webcam_detection(stream_url=url, conf=args.conf, save_output=args.save)


if __name__ == "__main__":
    main()
