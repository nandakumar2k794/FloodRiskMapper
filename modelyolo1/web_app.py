"""
Streamlit Web Dashboard for Land Waste & Object Detection.
Supports Custom Trash Model, General Objects Model, and Combined Hybrid Mode.

Usage:
    streamlit run web_app.py
"""
import time
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from detector import LandWasteDetector
from config import MODEL_PATH, COCO_MODEL_PATH, DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD, OUTPUT_DIR
from detect_ip_webcam import format_ip_url

# Page configuration
st.set_page_config(
    page_title="Land Waste & Object Vision AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2d3139;
        margin-bottom: 10px;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        height: 44px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector():
    """Load the YOLO detector engine and cache it."""
    return LandWasteDetector(trash_model_path=MODEL_PATH, coco_model_path=COCO_MODEL_PATH)


with st.spinner("Initializing YOLO Vision Engine..."):
    detector = load_detector()

# Sidebar Controls
st.sidebar.title("⚙️ Detection Settings")

# Model Detection Mode
detection_mode_choice = st.sidebar.selectbox(
    "AI Vision Mode",
    [
        "🌟 Combined (Trash + Normal Objects)",
        "🗑️ Land Waste & Trash Only",
        "📦 Normal Objects & Litter"
    ],
    index=0,
    help="Choose whether to detect both custom waste and normal objects or waste only."
)

mode_map = {
    "🌟 Combined (Trash + Normal Objects)": "hybrid",
    "🗑️ Land Waste & Trash Only": "trash",
    "📦 Normal Objects & Litter": "general"
}
active_mode = mode_map[detection_mode_choice]

# Confidence Slider (Default 0.15 for high sensitivity)
conf_thresh = st.sidebar.slider(
    "Detection Sensitivity (Confidence)",
    min_value=0.05,
    max_value=0.90,
    value=0.15,
    step=0.05,
    help="Lower value (e.g. 0.10 - 0.20) detects smaller or partially visible objects."
)

iou_thresh = st.sidebar.slider(
    "IOU / NMS Threshold",
    min_value=0.10,
    max_value=0.90,
    value=float(DEFAULT_IOU_THRESHOLD),
    step=0.05
)

st.sidebar.markdown("---")
input_mode = st.sidebar.radio(
    "Select Input Source",
    ["📱 Mobile IP Webcam (Live)", "📁 Upload Image", "🎥 Upload Video"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 How to connect Phone:
1. Open **IP Webcam** app on your phone.
2. Tap **Start server** at the bottom.
3. Enter the IP shown on phone (e.g. `192.168.1.15:8080`) in the box on the right.
4. Click **Start Stream**.
""")

# Main Page Header
st.title("🌍 Land Waste & Object Detection System")
st.caption("Real-Time AI Vision for Environmental Monitoring & Object Detection")

# ==========================================
# MODE 1: MOBILE IP WEBCAM STREAM
# ==========================================
if input_mode == "📱 Mobile IP Webcam (Live)":
    st.subheader("📱 Live Mobile Camera Stream")

    col_ip, col_btn1, col_btn2 = st.columns([3, 1, 1])

    with col_ip:
        ip_input = st.text_input(
            "Phone IP Webcam Address / URL:",
            value="http://192.168.1.10:8080/video",
            placeholder="e.g. 192.168.1.15:8080 or http://192.168.1.15:8080/video"
        )

    if "streaming" not in st.session_state:
        st.session_state.streaming = False

    with col_btn1:
        st.write("")
        st.write("")
        if st.button("▶ Start Stream", type="primary", use_container_width=True, disabled=st.session_state.streaming):
            st.session_state.streaming = True
            st.rerun()

    with col_btn2:
        st.write("")
        st.write("")
        if st.button("⏹ Stop Stream", type="secondary", use_container_width=True, disabled=not st.session_state.streaming):
            st.session_state.streaming = False
            st.rerun()

    col_video, col_stats = st.columns([3, 1])

    with col_video:
        video_placeholder = st.empty()

    with col_stats:
        status_placeholder = st.empty()
        count_placeholder = st.empty()
        fps_placeholder = st.empty()
        breakdown_placeholder = st.empty()

    if st.session_state.streaming:
        target_url = format_ip_url(ip_input)
        cap = cv2.VideoCapture(target_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            st.error(f"❌ Failed to connect to stream at: `{target_url}`. Check phone IP & ensure both devices are on same Wi-Fi.")
            st.session_state.streaming = False
        else:
            prev_time = time.time()
            fps = 0.0

            while st.session_state.streaming and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                curr_time = time.time()
                elapsed = curr_time - prev_time
                prev_time = curr_time
                if elapsed > 0:
                    fps = 0.85 * fps + 0.15 * (1.0 / elapsed)

                # Run YOLO detection with selected mode
                results = detector.detect(frame, conf=conf_thresh, iou=iou_thresh, mode=active_mode)
                annotated = results["annotated_image"]

                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

                # Update Stats
                sev_label = results["severity_label"]
                total_items = results["total_items"]
                class_counts = results["class_counts"]

                status_color = "#00e676" if "CLEAN" in sev_label else ("#ffab00" if "LOW" in sev_label else "#ff5252")

                status_placeholder.markdown(f"""
                <div class="metric-card">
                    <span style="color: #a0a0a0; font-size: 13px;">LAND CONDITION</span><br>
                    <span style="color: {status_color}; font-size: 20px; font-weight: bold;">{sev_label}</span>
                </div>
                """, unsafe_allow_html=True)

                count_placeholder.markdown(f"""
                <div class="metric-card">
                    <span style="color: #a0a0a0; font-size: 13px;">TOTAL DETECTED</span><br>
                    <span style="color: #00d2ff; font-size: 26px; font-weight: bold;">{total_items}</span>
                </div>
                """, unsafe_allow_html=True)

                fps_placeholder.markdown(f"""
                <div class="metric-card">
                    <span style="color: #a0a0a0; font-size: 13px;">LIVE SPEED</span><br>
                    <span style="color: #e0e0e0; font-size: 18px; font-weight: bold;">{fps:.1f} FPS</span>
                    <span style="color: #888; font-size: 12px;">({results['inference_time_ms']:.1f} ms)</span>
                </div>
                """, unsafe_allow_html=True)

                with breakdown_placeholder.container():
                    st.markdown("##### 📋 Identified Objects:")
                    if class_counts:
                        for name, cnt in class_counts.items():
                            st.write(f"• **{name}**: `{cnt}`")
                    else:
                        st.info("No objects / waste currently in view.")

            cap.release()
    else:
        video_placeholder.info("Enter your mobile IP Webcam address above and click **Start Stream** to begin live monitoring.")


# ==========================================
# MODE 2: UPLOAD STATIC IMAGE
# ==========================================
elif input_mode == "📁 Upload Image":
    st.subheader("📁 Land Image Analysis")
    uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png", "webp", "bmp"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        results = detector.detect(image, conf=conf_thresh, iou=iou_thresh, mode=active_mode)

        col_img, col_metrics = st.columns([3, 1])

        with col_img:
            annotated_rgb = cv2.cvtColor(results["annotated_image"], cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption=f"Detection Result - {uploaded_file.name}", use_container_width=True)

        with col_metrics:
            st.markdown(f"### Status: **{results['severity_label']}**")
            st.metric("Total Objects", results["total_items"])
            st.metric("Inference Speed", f"{results['inference_time_ms']:.1f} ms")

            st.markdown("#### Identified Objects")
            if results["class_counts"]:
                for name, cnt in results["class_counts"].items():
                    st.write(f"• **{name}**: `{cnt}`")
            else:
                st.success("Clean terrain! No objects detected.")

            res_jpg = cv2.imencode(".jpg", results["annotated_image"])[1].tobytes()
            st.download_button(
                label="💾 Download Annotated Image",
                data=res_jpg,
                file_name=f"detected_{uploaded_file.name}",
                mime="image/jpeg"
            )


# ==========================================
# MODE 3: UPLOAD VIDEO
# ==========================================
elif input_mode == "🎥 Upload Video":
    st.subheader("🎥 Land Survey Video Analysis")
    uploaded_video = st.file_uploader("Upload video file (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        temp_video_path = OUTPUT_DIR / f"temp_{uploaded_video.name}"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.info("Processing video frames with YOLO...")
        cap = cv2.VideoCapture(str(temp_video_path))
        video_view = st.empty()
        progress_bar = st.progress(0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % 2 == 0:
                res = detector.detect(frame, conf=conf_thresh, iou=iou_thresh, mode=active_mode)
                annotated_rgb = cv2.cvtColor(res["annotated_image"], cv2.COLOR_BGR2RGB)
                video_view.image(annotated_rgb, use_container_width=True)
                progress_bar.progress(min(1.0, frame_idx / total_frames))

        cap.release()
        st.success("Video processing completed!")
