# Land Waste & Object Detection System (YOLO11)

An automated AI vision system built using **YOLO11** to detect, identify, count, and assess waste and pollution on land in real-time.

---

## 📱 How to Stream from Mobile Phone to Laptop Screen

You can use your mobile phone camera as a wireless scanner to detect waste on land in real-time on your laptop screen.

### Step 1: Install IP Webcam App on Phone
1. On your Android phone, install **"IP Webcam"** from Google Play Store (or **DroidCam** / iOS equivalent).
2. Connect both your **Phone** and **Laptop** to the **same Wi-Fi network** (or connect your laptop to your phone's **Mobile Hotspot**).
3. Open the **IP Webcam** app on your phone, scroll to the bottom, and tap **"Start server"**.
4. The app will display an IP address on your phone screen, for example:
   ```
   http://192.168.1.15:8080
   ```

---

### Step 2: Run Real-Time Detection on Laptop

#### Method A: Interactive Desktop GUI (Easiest)
```powershell
python gui_app.py
```
1. Paste your phone's IP address (e.g. `http://192.168.1.15:8080/video`) in the input box.
2. Click **"▶ Connect Mobile Stream"**.
3. The live camera feed with real-time detection boxes, waste counts, and pollution status will display directly on your screen!

#### Method B: Standalone High-Speed Window
```powershell
python detect_ip_webcam.py --url http://192.168.1.15:8080/video
```
* **Hotkeys in Live Stream**:
  * `f` : Toggle Fullscreen mode
  * `s` : Save Snapshot of current detected frame
  * `+` / `-` : Increase / decrease detection confidence threshold
  * `h` : Toggle HUD overlay
  * `q` / `ESC` : Exit stream

---

## 📁 Project Structure

```
modelyolo/
│
├── yolo11_trash_detection.pt   # Pretrained YOLO11 weights
├── config.py                   # Configuration (classes, colors, severity thresholds)
├── detector.py                 # Core LandWasteDetector class & visual HUD engine
├── detect_ip_webcam.py         # Mobile Phone IP Webcam live stream detection
├── gui_app.py                  # Interactive Desktop GUI (Images + Mobile Stream)
├── detect_image.py             # Single / Batch image inference script
├── detect_video.py             # Video & Drone footage processing script
├── detect_webcam.py            # Built-in laptop camera detection script
├── main.py                     # All-in-one launcher
├── requirements.txt            # Python dependencies
└── runs/detect/                # Auto-created output directory for saved results
```

---

## 💻 Other Usage Modes

### 1. Detect on Static Images
```powershell
python detect_image.py --source path/to/land_image.jpg --conf 0.25 --show
```

### 2. Detect on Land Survey Videos
```powershell
python detect_video.py --source path/to/drone_survey.mp4 --conf 0.25 --show
```

### 3. Built-in Laptop Webcam
```powershell
python detect_webcam.py --camera 0
```
