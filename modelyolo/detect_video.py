"""
Land Waste Detection on Video / Drone Footage.
Usage:
    python detect_video.py --source path/to/drone_land_survey.mp4
"""
import argparse
import time
from pathlib import Path
import cv2
from detector import LandWasteDetector
from config import OUTPUT_DIR, DEFAULT_CONF_THRESHOLD


def process_video(source_path: Path, output_dir: Path, conf: float, show: bool = False):
    detector = LandWasteDetector(conf=conf)

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        print(f"[!] Error: Could not open video file at '{source_path}'")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = output_dir / f"detected_{source_path.name}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"[*] Processing Video: {source_path.name} ({total_frames} frames @ {fps:.1f} FPS)...")

    frame_idx = 0
    start_total_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            results = detector.detect(frame, conf=conf)
            annotated_frame = results["annotated_image"]

            out.write(annotated_frame)

            if frame_idx % 30 == 0 or frame_idx == total_frames:
                elapsed = time.time() - start_total_time
                current_fps = frame_idx / elapsed if elapsed > 0 else 0
                progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                print(f"    Progress: {frame_idx}/{total_frames} ({progress:.1f}%) | Processing FPS: {current_fps:.1f} | Detections: {results['total_items']}")

            if show:
                cv2.imshow("Land Waste Detection - Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[!] Video playback stopped by user.")
                    break

    finally:
        cap.release()
        out.release()
        if show:
            cv2.destroyAllWindows()

    print(f"\n[+] Video processing complete! Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Process video footage for land waste detection.")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to video file (e.g., .mp4, .avi, .mov).")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD,
                        help=f"Confidence threshold (default: {DEFAULT_CONF_THRESHOLD})")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help="Output directory to save the annotated video.")
    parser.add_argument("--show", action="store_true",
                        help="Display video frames during processing.")

    args = parser.parse_args()
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"[!] Error: Video file '{source_path}' does not exist.")
        return

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    process_video(source_path, out_dir, args.conf, args.show)


if __name__ == "__main__":
    main()
