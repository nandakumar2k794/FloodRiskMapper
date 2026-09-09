"""
Land Waste & Object Detection - Interactive Desktop GUI App.
Supports Static Images & Live Mobile IP Webcam / Camera Streaming.
"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import threading
import time
import cv2
from PIL import Image, ImageTk

from detector import LandWasteDetector
from config import MODEL_PATH, DEFAULT_CONF_THRESHOLD, OUTPUT_DIR
from detect_ip_webcam import format_ip_url


class WasteDetectionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Land Waste & Object Detection (YOLO11) - Mobile & Image")
        self.root.geometry("1150x780")
        self.root.minsize(950, 650)
        self.root.configure(bg="#1e1e2e")

        self.detector = None
        self.current_image_path = None
        self.last_result = None
        self.photo_image = None

        # Live Stream controls
        self.is_streaming = False
        self.stream_thread = None
        self.cap = None

        self._build_ui()
        self._load_model_async()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # Top Header Bar
        header = tk.Frame(self.root, bg="#181825", height=60)
        header.pack(fill=tk.X, side=tk.TOP)

        title = tk.Label(
            header,
            text="Land Waste & Object Vision System",
            font=("Segoe UI", 16, "bold"),
            bg="#181825",
            fg="#cdd6f4"
        )
        title.pack(side=tk.LEFT, padx=20, pady=12)

        self.model_status = tk.Label(
            header,
            text="Loading YOLO11 model...",
            font=("Segoe UI", 10),
            bg="#181825",
            fg="#f9e2af"
        )
        self.model_status.pack(side=tk.RIGHT, padx=20, pady=12)

        # Main Layout: Left Control Panel + Right Image Display
        main_container = tk.Frame(self.root, bg="#1e1e2e")
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Left Panel (Controls & Stats)
        left_panel = tk.Frame(main_container, bg="#181825", width=340, padx=15, pady=15)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left_panel.pack_propagate(False)

        # Mode Selection Notebook / Tabs
        lbl_section1 = tk.Label(left_panel, text="Input Sources:", font=("Segoe UI", 11, "bold"), bg="#181825", fg="#cdd6f4")
        lbl_section1.pack(anchor="w", pady=(0, 6))

        # Static Image Button
        btn_open = tk.Button(
            left_panel,
            text="📁 Open Image File",
            command=self.open_image,
            font=("Segoe UI", 10, "bold"),
            bg="#89b4fa",
            fg="#11111b",
            relief=tk.FLAT,
            cursor="hand2",
            pady=6
        )
        btn_open.pack(fill=tk.X, pady=(0, 8))

        # Mobile IP Webcam Section
        tk.Label(left_panel, text="📱 Mobile IP Camera URL:", font=("Segoe UI", 9), bg="#181825", fg="#a6adc8").pack(anchor="w")
        self.entry_ip = tk.Entry(left_panel, font=("Consolas", 9), bg="#313244", fg="#cdd6f4", insertbackground="white", relief=tk.FLAT)
        self.entry_ip.insert(0, "http://192.168.1.10:8080/video")
        self.entry_ip.pack(fill=tk.X, pady=(2, 6))

        self.btn_stream = tk.Button(
            left_panel,
            text="▶ Connect Mobile Stream",
            command=self.toggle_stream,
            font=("Segoe UI", 10, "bold"),
            bg="#a6e3a1",
            fg="#11111b",
            relief=tk.FLAT,
            cursor="hand2",
            pady=6
        )
        self.btn_stream.pack(fill=tk.X, pady=(0, 10))

        # Confidence Slider
        lbl_conf = tk.Label(left_panel, text="Detection Confidence:", font=("Segoe UI", 9, "bold"), bg="#181825", fg="#cdd6f4")
        lbl_conf.pack(anchor="w", pady=(5, 2))

        self.conf_val_label = tk.Label(left_panel, text=f"{DEFAULT_CONF_THRESHOLD:.2f}", font=("Segoe UI", 9, "bold"), bg="#181825", fg="#89dceb")
        self.conf_val_label.pack(anchor="w")

        self.slider_conf = ttk.Scale(
            left_panel,
            from_=0.05,
            to=0.95,
            value=DEFAULT_CONF_THRESHOLD,
            orient=tk.HORIZONTAL,
            command=self._on_slider_change
        )
        self.slider_conf.pack(fill=tk.X, pady=(0, 10))

        # Action Buttons
        btn_save = tk.Button(
            left_panel,
            text="💾 Save Annotated Image / Snapshot",
            command=self.save_result,
            font=("Segoe UI", 9),
            bg="#313244",
            fg="#cdd6f4",
            relief=tk.FLAT,
            cursor="hand2",
            pady=6
        )
        btn_save.pack(fill=tk.X, pady=(0, 10))

        # Divider
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # Statistics Summary
        tk.Label(left_panel, text="Land Analytics & Status:", font=("Segoe UI", 11, "bold"), bg="#181825", fg="#cdd6f4").pack(anchor="w", pady=(2, 4))

        self.lbl_status = tk.Label(left_panel, text="Status: Ready", font=("Segoe UI", 10, "bold"), bg="#181825", fg="#a6adc8")
        self.lbl_status.pack(anchor="w", pady=1)

        self.lbl_count = tk.Label(left_panel, text="Total Objects: 0", font=("Segoe UI", 10), bg="#181825", fg="#a6adc8")
        self.lbl_count.pack(anchor="w", pady=1)

        self.lbl_speed = tk.Label(left_panel, text="Speed / FPS: --", font=("Segoe UI", 10), bg="#181825", fg="#a6adc8")
        self.lbl_speed.pack(anchor="w", pady=1)

        tk.Label(left_panel, text="Detected Waste Items:", font=("Segoe UI", 9, "bold"), bg="#181825", fg="#cdd6f4").pack(anchor="w", pady=(8, 2))

        # Class Breakdown Box
        self.txt_breakdown = tk.Text(left_panel, height=7, bg="#11111b", fg="#cdd6f4", font=("Consolas", 9), relief=tk.FLAT)
        self.txt_breakdown.pack(fill=tk.BOTH, expand=True, pady=(2, 0))
        self.txt_breakdown.config(state=tk.DISABLED)

        # Right Panel (Image Display)
        self.right_panel = tk.Frame(main_container, bg="#11111b")
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Label(
            self.right_panel,
            bg="#11111b",
            text="Open an image or connect Mobile IP Webcam to view live detection",
            fg="#6c7086",
            font=("Segoe UI", 12)
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def _load_model_async(self):
        def _load():
            try:
                self.detector = LandWasteDetector(model_path=MODEL_PATH)
                self.model_status.config(text="Model: Ready (YOLO11)", fg="#a6e3a1")
            except Exception as e:
                self.model_status.config(text=f"Model Error: {e}", fg="#f38ba8")
        threading.Thread(target=_load, daemon=True).start()

    def _on_slider_change(self, val):
        self.conf_val_label.config(text=f"{float(val):.2f}")
        if not self.is_streaming and self.current_image_path:
            self.run_detection()

    def open_image(self):
        if self.is_streaming:
            self.stop_stream()

        file_path = filedialog.askopenfilename(
            title="Select Land Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff")]
        )
        if file_path:
            self.current_image_path = file_path
            self.run_detection()

    def run_detection(self):
        if not self.current_image_path:
            return
        if not self.detector:
            messagebox.showwarning("Warning", "Model is still loading. Please wait a moment.")
            return

        conf = float(self.slider_conf.get())
        self.last_result = self.detector.detect(self.current_image_path, conf=conf)
        self._update_stats_ui(self.last_result)
        self._display_image(self.last_result['annotated_image'])

    def toggle_stream(self):
        if self.is_streaming:
            self.stop_stream()
        else:
            self.start_stream()

    def start_stream(self):
        if not self.detector:
            messagebox.showwarning("Warning", "Model is still loading. Please wait.")
            return

        url_input = self.entry_ip.get().strip()
        if not url_input:
            messagebox.showerror("Error", "Please enter the IP Webcam URL shown on your phone.")
            return

        stream_url = format_ip_url(url_input)
        self.btn_stream.config(text="Connecting...", bg="#f9e2af", state=tk.DISABLED)
        self.root.update()

        def _connect_and_stream():
            self.cap = cv2.VideoCapture(stream_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                self.root.after(0, lambda: self._stream_failed(stream_url))
                return

            self.is_streaming = True
            self.root.after(0, lambda: self.btn_stream.config(
                text="⏹ Stop Stream", bg="#f38ba8", state=tk.NORMAL
            ))

            prev_time = time.time()
            fps_smooth = 0.0

            while self.is_streaming and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                curr_time = time.time()
                elapsed = curr_time - prev_time
                prev_time = curr_time
                if elapsed > 0:
                    fps_smooth = 0.85 * fps_smooth + 0.15 * (1.0 / elapsed)

                conf = float(self.slider_conf.get())
                result = self.detector.detect(frame, conf=conf)
                self.last_result = result

                # Update UI in main thread
                annotated = result['annotated_image']
                self.root.after(0, lambda r=result, f=fps_smooth, img=annotated: self._update_stream_ui(r, f, img))

            if self.cap:
                self.cap.release()

        self.stream_thread = threading.Thread(target=_connect_and_stream, daemon=True)
        self.stream_thread.start()

    def _stream_failed(self, url):
        self.btn_stream.config(text="▶ Connect Mobile Stream", bg="#a6e3a1", state=tk.NORMAL)
        messagebox.showerror(
            "Connection Failed",
            f"Could not connect to:\n{url}\n\n"
            "Please check:\n"
            "1. 'IP Webcam' app server is running on your phone.\n"
            "2. Both laptop and phone are on the same Wi-Fi / Hotspot network.\n"
            "3. IP address and port match exactly."
        )

    def stop_stream(self):
        self.is_streaming = False
        if self.cap:
            self.cap.release()
        self.btn_stream.config(text="▶ Connect Mobile Stream", bg="#a6e3a1", state=tk.NORMAL)
        self.lbl_speed.config(text="Stream Stopped")

    def _update_stream_ui(self, res, fps, annotated_img):
        if not self.is_streaming:
            return
        self.lbl_status.config(
            text=f"Status: {res['severity_label']}",
            fg="#a6e3a1" if "CLEAN" in res['severity_label'] else "#fab387"
        )
        self.lbl_count.config(text=f"Total Objects: {res['total_items']}", fg="#cdd6f4")
        self.lbl_speed.config(text=f"Live FPS: {fps:.1f} ({res['inference_time_ms']:.1f}ms)", fg="#cdd6f4")

        # Class counts
        self.txt_breakdown.config(state=tk.NORMAL)
        self.txt_breakdown.delete("1.0", tk.END)
        if res['class_counts']:
            for name, count in res['class_counts'].items():
                self.txt_breakdown.insert(tk.END, f"• {name}: {count}\n")
        else:
            self.txt_breakdown.insert(tk.END, "No waste detected in field of view.\n")
        self.txt_breakdown.config(state=tk.DISABLED)

        self._display_image(annotated_img)

    def _update_stats_ui(self, res):
        self.lbl_status.config(
            text=f"Status: {res['severity_label']}",
            fg="#a6e3a1" if "CLEAN" in res['severity_label'] else "#fab387"
        )
        self.lbl_count.config(text=f"Total Objects: {res['total_items']}", fg="#cdd6f4")
        self.lbl_speed.config(text=f"Speed: {res['inference_time_ms']:.1f} ms", fg="#cdd6f4")

        self.txt_breakdown.config(state=tk.NORMAL)
        self.txt_breakdown.delete("1.0", tk.END)
        if res['class_counts']:
            for name, count in res['class_counts'].items():
                self.txt_breakdown.insert(tk.END, f"• {name}: {count}\n")
        else:
            self.txt_breakdown.insert(tk.END, "No waste detected.\n")
        self.txt_breakdown.config(state=tk.DISABLED)

    def _display_image(self, bgr_img):
        if bgr_img is None:
            return
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        panel_w = max(400, self.right_panel.winfo_width())
        panel_h = max(400, self.right_panel.winfo_height())
        pil_img.thumbnail((panel_w, panel_h), Image.Resampling.BILINEAR)

        self.photo_image = ImageTk.PhotoImage(pil_img)
        self.canvas.config(image=self.photo_image, text="")

    def save_result(self):
        if not self.last_result:
            messagebox.showinfo("Notice", "No detection image to save.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png")],
            initialdir=str(OUTPUT_DIR),
            initialfile=f"detected_snapshot_{int(time.time())}.jpg"
        )
        if save_path:
            cv2.imwrite(save_path, self.last_result['annotated_image'])
            messagebox.showinfo("Saved", f"Annotated image saved successfully to:\n{save_path}")

    def _on_close(self):
        self.is_streaming = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = WasteDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
