"""
Land Waste Detection on Images.
Usage:
    python detect_image.py --source path/to/image.jpg
    python detect_image.py --source path/to/images_folder/ --conf 0.3
"""
import argparse
from pathlib import Path
import cv2
from detector import LandWasteDetector
from config import OUTPUT_DIR, DEFAULT_CONF_THRESHOLD

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def process_image(detector: LandWasteDetector, img_path: Path, output_dir: Path, conf: float, show: bool = False):
    print(f"\n[>] Processing: {img_path.name}")
    results = detector.detect(img_path, conf=conf)

    # Print summary to console
    print(f"    - Status: {results['severity_label']}")
    print(f"    - Total Objects Detected: {results['total_items']}")
    print(f"    - Inference Speed: {results['inference_time_ms']:.1f} ms")
    if results['class_counts']:
        print("    - Breakdown:")
        for cls_name, count in results['class_counts'].items():
            print(f"        * {cls_name}: {count}")
    else:
        print("    - No waste detected.")

    # Save output
    output_path = output_dir / f"detected_{img_path.name}"
    cv2.imwrite(str(output_path), results['annotated_image'])
    print(f"    [+] Saved annotated result to: {output_path}")

    if show:
        cv2.imshow(f"Detection - {img_path.name}", results['annotated_image'])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Detect waste and objects on land in images.")
    parser.add_argument("--source", type=str, required=False, default=None,
                        help="Path to an image file or directory of images.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD,
                        help=f"Confidence threshold (default: {DEFAULT_CONF_THRESHOLD})")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help="Output directory to save results.")
    parser.add_argument("--show", action="store_true",
                        help="Display the results in a window.")

    args = parser.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = LandWasteDetector(conf=args.conf)

    if args.source is None:
        print("\n[!] No --source specified.")
        print("[*] Example: python detect_image.py --source my_land_photo.jpg")
        return

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"[!] Error: Source path '{source_path}' does not exist.")
        return

    if source_path.is_file():
        if source_path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            process_image(detector, source_path, out_dir, args.conf, args.show)
        else:
            print(f"[!] Unsupported image format: {source_path.suffix}")
    elif source_path.is_dir():
        image_files = [p for p in source_path.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
        if not image_files:
            print(f"[!] No supported images found in '{source_path}'.")
            return
        print(f"[*] Found {len(image_files)} images in directory. Starting batch processing...")
        for img_file in image_files:
            process_image(detector, img_file, out_dir, args.conf, args.show)
        print(f"\n[+] Batch processing finished. All results saved to: {out_dir}")


if __name__ == "__main__":
    main()
