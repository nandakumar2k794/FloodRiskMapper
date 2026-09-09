"""
Real-time Land Waste Detection via Camera / Webcam.
Usage:
    python detect_webcam.py
    python detect_webcam.py --camera 0 --conf 0.3
"""
import argparse
import time
import cv2
from detector import LandWasteDetector
from config import DEFAULT_CONF_THRESHOLD


def run_webcam(camera_index: int = 0, conf: float = DEFAULT_CONF_THRESHOLD):
    print(f"[*] Starting Land Waste Detection on camera index {camera_index}...")
    detector = LandWasteDetector(conf=conf)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[!] Error: Could not access camera with index {camera_index}.")
        return

    # Attempt to set HD resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[+] Camera stream active.")
    print("[*] Press 'q' or 'ESC' to exit.")

    prev_time = time.time()
    fps_display = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Failed to grab frame from camera.")
                break

            # Calculate FPS
            curr_time = time.time()
            fps_display = 0.9 * fps_display + 0.1 * (1.0 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else 30.0
            prev_time = curr_time

            # Run detection
            results = detector.detect(frame, conf=conf)
            annotated = results["annotated_image"]

            # Add FPS counter to display
            cv2.putText(
                annotated,
                f"FPS: {fps_display:.1f}",
                (annotated.shape[1] - 140, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("Real-Time Land Waste Detection (YOLO11)", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 is ESC key
                print("[*] Exiting camera feed.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-time Land Waste Detection via Camera.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD,
                        help=f"Confidence threshold (default: {DEFAULT_CONF_THRESHOLD})")
    args = parser.parse_args()

    run_webcam(camera_index=args.camera, conf=args.conf)


if __name__ == "__main__":
    main()
