"""
Main entry point for Land Waste & Object Detection System.
Provides a unified launcher for:
1. Modern Web Dashboard (Streamlit - Browser UI)
2. Native Desktop GUI (Tkinter)
3. Live Mobile IP Webcam Window (OpenCV)
4. Static Image / Video Detection CLI
"""
import sys
import subprocess
import argparse


def show_menu():
    print("=" * 65)
    print("   🌍 LAND WASTE & OBJECT DETECTION SYSTEM (YOLO11)")
    print("=" * 65)
    print("Choose your preferred UI / Mode:")
    print("  1. 🌐 Modern Web Dashboard (Streamlit - Browser UI)")
    print("  2. 🖥️ Native Desktop GUI (Tkinter Window)")
    print("  3. 📱 Direct Mobile IP Webcam Stream (High Speed Window)")
    print("  4. 📁 Detect on Static Image(s)")
    print("  5. 🎥 Detect on Video File (e.g. Drone Survey)")
    print("  6. 💻 Detect via Built-in Laptop Webcam")
    print("  7. ❌ Exit")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Land Waste Detection Launcher")
    parser.add_argument("--mode", choices=["web", "gui", "mobile", "image", "video", "webcam"], default=None)
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.25)

    args, unknown = parser.parse_known_args()
    mode = args.mode

    if mode is None:
        show_menu()
        choice = input("Enter choice (1-7): ").strip()
        if choice == "1":
            mode = "web"
        elif choice == "2":
            mode = "gui"
        elif choice == "3":
            mode = "mobile"
        elif choice == "4":
            mode = "image"
        elif choice == "5":
            mode = "video"
        elif choice == "6":
            mode = "webcam"
        else:
            print("[*] Exiting.")
            sys.exit(0)

    if mode == "web":
        print("[*] Launching Streamlit Web Dashboard in your browser...")
        subprocess.run(["streamlit", "run", "web_app.py"])
    elif mode == "gui":
        import gui_app
        gui_app.main()
    elif mode == "mobile":
        import detect_ip_webcam
        url = args.url or input("Enter phone IP Webcam URL (e.g. http://192.168.1.10:8080): ").strip()
        detect_ip_webcam.run_ip_webcam_detection(stream_url=url, conf=args.conf)
    elif mode == "image":
        import detect_image
        if not args.source:
            args.source = input("Enter path to image file or directory: ").strip().strip('"')
        sys.argv = ["detect_image.py", "--source", args.source, "--conf", str(args.conf)]
        detect_image.main()
    elif mode == "video":
        import detect_video
        if not args.source:
            args.source = input("Enter path to video file: ").strip().strip('"')
        sys.argv = ["detect_video.py", "--source", args.source, "--conf", str(args.conf)]
        detect_video.main()
    elif mode == "webcam":
        import detect_webcam
        detect_webcam.main()


if __name__ == "__main__":
    main()
