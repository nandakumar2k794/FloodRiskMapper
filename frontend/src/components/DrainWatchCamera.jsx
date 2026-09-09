import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Wifi,
  Camera,
  Image as ImageIcon,
  Video,
  RefreshCw,
  Play,
  Square,
  Upload,
  Search,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  Flame,
  Layers,
  Sparkles,
  Info,
  HelpCircle,
  ExternalLink,
  Usb
} from 'lucide-react';
import {
  detectFrame,
  detectPhoneStream,
  fetchPhoneTelemetry,
  stopPhoneStream,
  uploadMedia,
  discoverPhoneCamera
} from '../api/client';

// USB-C API helpers (call backend endpoints directly)
const API_BASE = 'http://127.0.0.1:8000/api/v1';
const usbcFeedUrl = (deviceIndex) => `${API_BASE}/drainwatch/usbc-feed?device_index=${deviceIndex}`;
const fetchUsbcTelemetry = () => fetch(`${API_BASE}/drainwatch/usbc-telemetry`).then(r => r.json());
const stopUsbcStream = () => fetch(`${API_BASE}/drainwatch/usbc-stop`, { method: 'POST' });

// 8 Custom Trash Classes with designated color codes matching modelyolo config
const TRASH_CLASS_COLORS = {
  'Plastic / Bottle': '#ff9f00',
  'Paper / Cardboard': '#38bdf8',
  'Metal / Can': '#eab308',
  'Glass': '#ec4899',
  'Organic / Debris': '#22c55e',
  'Polythene / Bag': '#f97316',
  'E-Waste / Hazardous': '#ef4444',
  'General Land Waste': '#a855f7',
};

// Helper: Ensure 4-octet IPv4 or clean URL
const normalizeStreamUrl = (raw) => {
  let url = (raw || '').trim();
  if (!url) return 'http://192.168.137.1:8080/video';
  if (!url.startsWith('http://') && !url.startsWith('https://')) {
    url = `http://${url}`;
  }
  try {
    const parsed = new URL(url);
    let host = parsed.hostname || '192.168.137.1';
    let port = parsed.port;
    const parts = host.split('.');
    if (parts.length === 3 && parts.every((p) => /^\d+$/.test(p))) {
      host = `${host}.1`;
    }
    if (!port) {
      port = url.includes('4747') ? '4747' : '8080';
    }
    return `http://${host}:${port}/video`;
  } catch {
    return url;
  }
};

export default function DrainWatchCamera({ station, onTelemetryUpdate }) {
  // 5 Modes:
  // 1. 'mobile'  -> IP Webcam / Phone Stream (MJPEG)
  // 2. 'webcam'  -> Laptop / USB Webcam (browser getUserMedia)
  // 3. 'usbc'    -> USB-C External Camera (backend MJPEG stream)
  // 4. 'image'   -> Static Image File Detection
  // 5. 'video'   -> Video Survey Detection
  const [activeMode, setActiveMode] = useState('mobile');

  // Detection Parameters (matching modelyolo/config.py)
  const [confThreshold, setConfThreshold] = useState(0.15);

  // ── Mode 1: Mobile IP Webcam States ─────────────────────────────────────────
  const [ipUrl, setIpUrl] = useState('http://192.168.137.1:8080/video');
  const [isIpStreaming, setIsIpStreaming] = useState(false);
  const [streamKey, setStreamKey] = useState(Date.now());
  const [streamFps, setStreamFps] = useState(0);
  const [isAutoDiscovering, setIsAutoDiscovering] = useState(false);
  const [streamOffline, setStreamOffline] = useState(false);

  // ── Mode 2: Laptop / USB Webcam States ──────────────────────────────────────
  const [videoDevices, setVideoDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');
  const [isWebcamActive, setIsWebcamActive] = useState(false);

  // ── Mode 3: USB-C External Camera States ────────────────────────────────────
  const [usbcDeviceIndex, setUsbcDeviceIndex] = useState(0);
  const [isUsbcStreaming, setIsUsbcStreaming] = useState(false);
  const [usbcStreamKey, setUsbcStreamKey] = useState(Date.now());
  const [usbcFps, setUsbcFps] = useState(0);
  const [usbcOffline, setUsbcOffline] = useState(false);

  // ── Mode 4: Image File Detection States ─────────────────────────────────────
  const [uploadedImageName, setUploadedImageName] = useState('');
  const [uploadedImagePreview, setUploadedImagePreview] = useState(null);

  // ── Mode 5: Video Survey States ──────────────────────────────────────────────
  const [uploadedVideoName, setUploadedVideoName] = useState('');
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState(null);
  const [isVideoAnalyzing, setIsVideoAnalyzing] = useState(false);

  // Detection Results & HUD
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionResult, setDetectionResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');

  // Refs
  const imageInputRef = useRef(null);
  const videoInputRef = useRef(null);
  const phoneNativeCameraRef = useRef(null);
  const webcamVideoRef = useRef(null);
  const webcamCanvasRef = useRef(null);
  const webcamStreamRef = useRef(null);
  const webcamIntervalRef = useRef(null);
  const telemetryIntervalRef = useRef(null);
  const usbcTelemetryIntervalRef = useRef(null);
  const ipImgRef = useRef(null);
  const ipRefreshIntervalRef = useRef(null);
  const videoSurveyPlayerRef = useRef(null);
  const videoSurveyCanvasRef = useRef(null);
  const videoSurveyIntervalRef = useRef(null);

  // Enumerate Available Video Devices
  const enumerateCameras = async () => {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = devices.filter((d) => d.kind === 'videoinput');
      setVideoDevices(videoInputs);
      if (videoInputs.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoInputs[0].deviceId);
      }
    } catch (err) {
      console.debug('Camera enumeration error:', err);
    }
  };

  useEffect(() => {
    enumerateCameras();
  }, [activeMode]);

  // Clean up on unmount or tab switch
  const stopAllActiveFeeds = () => {
    // Stop IP stream
    if (isIpStreaming) {
      setIsIpStreaming(false);
      setStreamOffline(false);
      stopPhoneStream().catch(() => {});
    }
    // Clear IP stream refresh interval
    if (ipRefreshIntervalRef.current) {
      clearInterval(ipRefreshIntervalRef.current);
      ipRefreshIntervalRef.current = null;
    }
    // Stop Webcam stream
    if (webcamStreamRef.current) {
      webcamStreamRef.current.getTracks().forEach((t) => t.stop());
      webcamStreamRef.current = null;
    }
    if (webcamIntervalRef.current) {
      clearInterval(webcamIntervalRef.current);
      webcamIntervalRef.current = null;
    }
    setIsWebcamActive(false);

    // Stop USB-C stream
    if (isUsbcStreaming) {
      setIsUsbcStreaming(false);
      setUsbcOffline(false);
      stopUsbcStream().catch(() => {});
    }
    if (usbcTelemetryIntervalRef.current) {
      clearInterval(usbcTelemetryIntervalRef.current);
      usbcTelemetryIntervalRef.current = null;
    }

    // Stop Video Survey
    if (videoSurveyIntervalRef.current) {
      clearInterval(videoSurveyIntervalRef.current);
      videoSurveyIntervalRef.current = null;
    }
    setIsVideoAnalyzing(false);

    // Clear telemetry polling
    if (telemetryIntervalRef.current) {
      clearInterval(telemetryIntervalRef.current);
      telemetryIntervalRef.current = null;
    }
  };

  useEffect(() => {
    return () => {
      stopAllActiveFeeds();
    };
  }, []);

  const handleModeChange = (mode) => {
    stopAllActiveFeeds();
    setErrorMessage('');
    setActiveMode(mode);
  };

  // ───────────────────────────────────────────────────────────────────────────
  // Mode 1: Mobile IP Webcam Handlers (detect_ip_webcam.py)
  // ───────────────────────────────────────────────────────────────────────────
  const handleAutoDiscover = async () => {
    try {
      setIsAutoDiscovering(true);
      setErrorMessage('');
      const res = await discoverPhoneCamera();
      if (res?.active_cameras && res.active_cameras.length > 0) {
        const found = res.active_cameras[0].url;
        setIpUrl(found);
        setErrorMessage(`✅ Found active phone camera at ${found}! Connecting...`);
        setIsIpStreaming(true);
        setStreamOffline(false);
        setStreamKey(Date.now());
      } else {
        setErrorMessage(
          'No camera server automatically found on network. Please open IP Webcam (:8080) or DroidCam (:4747) on your phone, click "Start server", and type the IP shown on your phone.'
        );
      }
    } catch (err) {
      console.error('Auto discover error:', err);
      setErrorMessage('Discovery probe failed. Please check phone Wi-Fi connection.');
    } finally {
      setIsAutoDiscovering(false);
    }
  };

  const handleIpScanSingleFrame = async () => {
    const cleanUrl = normalizeStreamUrl(ipUrl);
    setIpUrl(cleanUrl);

    try {
      setIsDetecting(true);
      setErrorMessage('');
      const res = await detectPhoneStream(cleanUrl, station?.camera_id || 'CAM_VELACHERY_01', confThreshold);
      setDetectionResult(res);
      setStreamOffline(false);
      if (onTelemetryUpdate) onTelemetryUpdate(res);
    } catch (err) {
      console.error('IP stream scan error:', err);
      setStreamOffline(true);
      setErrorMessage(
        `Could not reach phone camera at ${cleanUrl}. 1) Ensure IP Webcam / DroidCam is active on phone, 2) Verify phone & PC are on the same Wi-Fi, 3) Type the exact IP shown on your phone.`
      );
    } finally {
      setIsDetecting(false);
    }
  };

  const handleStartIpStream = () => {
    const cleanUrl = normalizeStreamUrl(ipUrl);
    setIpUrl(cleanUrl);
    setErrorMessage('');
    setStreamOffline(false);
    const key = Date.now();
    setStreamKey(key);
    setIsIpStreaming(true);
  };

  const handleStopIpStream = async () => {
    setIsIpStreaming(false);
    setStreamOffline(false);
    if (ipRefreshIntervalRef.current) {
      clearInterval(ipRefreshIntervalRef.current);
      ipRefreshIntervalRef.current = null;
    }
    if (telemetryIntervalRef.current) {
      clearInterval(telemetryIntervalRef.current);
      telemetryIntervalRef.current = null;
    }
    try {
      await stopPhoneStream();
    } catch (e) {
      console.debug('Stop stream notice:', e);
    }
  };

  // Periodically bust cache on MJPEG img src to prevent single-frame freeze
  useEffect(() => {
    if (isIpStreaming && ipImgRef.current) {
      // Force the browser to re-request the MJPEG stream every 80ms
      // This keeps the stream alive even if browser caches the first frame
      ipRefreshIntervalRef.current = setInterval(() => {
        if (ipImgRef.current && !streamOffline) {
          const base = `${API_BASE}/drainwatch/video-feed?stream_url=${encodeURIComponent(ipUrl)}`;
          ipImgRef.current.src = `${base}&_t=${Date.now()}`;
        }
      }, 80);
    } else {
      if (ipRefreshIntervalRef.current) {
        clearInterval(ipRefreshIntervalRef.current);
        ipRefreshIntervalRef.current = null;
      }
    }
    return () => {
      if (ipRefreshIntervalRef.current) clearInterval(ipRefreshIntervalRef.current);
    };
  }, [isIpStreaming, ipUrl, streamOffline]);

  // Telemetry polling for live IP stream
  useEffect(() => {
    if (isIpStreaming) {
      const poll = async () => {
        try {
          const res = await fetch(`http://127.0.0.1:8000/api/v1/drainwatch/phone-telemetry?conf_threshold=${confThreshold}`);
          const telem = await res.json();
          if (telem) {
            if (telem.fps !== undefined) setStreamFps(telem.fps);
            if (telem.error) {
              setErrorMessage(telem.error);
              setStreamOffline(true);
            } else if (telem.total_items !== undefined) {
              setStreamOffline(false);
              setDetectionResult(telem);
              if (onTelemetryUpdate) onTelemetryUpdate(telem);
            }
          }
        } catch (e) {
          console.debug('Telemetry poll error:', e);
        }
      };
      poll();
      telemetryIntervalRef.current = setInterval(poll, 1200);
    } else {
      if (telemetryIntervalRef.current) {
        clearInterval(telemetryIntervalRef.current);
        telemetryIntervalRef.current = null;
      }
    }
    return () => {
      if (telemetryIntervalRef.current) clearInterval(telemetryIntervalRef.current);
    };
  }, [isIpStreaming, confThreshold]);

  // ───────────────────────────────────────────────────────────────────────────
  // Mode 3: USB-C External Camera Handlers
  // ───────────────────────────────────────────────────────────────────────────
  const handleStartUsbcStream = () => {
    setErrorMessage('');
    setUsbcOffline(false);
    setUsbcStreamKey(Date.now());
    setIsUsbcStreaming(true);
  };

  const handleStopUsbcStream = async () => {
    setIsUsbcStreaming(false);
    setUsbcOffline(false);
    if (usbcTelemetryIntervalRef.current) {
      clearInterval(usbcTelemetryIntervalRef.current);
      usbcTelemetryIntervalRef.current = null;
    }
    try { await stopUsbcStream(); } catch (e) { console.debug('USB-C stop:', e); }
  };

  // USB-C telemetry polling
  const usbcErrorCountRef = useRef(0);

  useEffect(() => {
    if (isUsbcStreaming) {
      usbcErrorCountRef.current = 0;
      const poll = async () => {
        try {
          const res = await fetch(`http://127.0.0.1:8000/api/v1/drainwatch/usbc-telemetry?conf_threshold=${confThreshold}`);
          const telem = await res.json();
          usbcErrorCountRef.current = 0; // reset on success
          if (telem) {
            if (telem.fps !== undefined) setUsbcFps(telem.fps);
            if (telem.status === 'DEVICE_NOT_FOUND') {
              setUsbcOffline(true);
              setErrorMessage(`USB-C device ${usbcDeviceIndex} not found. Check cable and try a different device index.`);
            } else if (telem.total_items !== undefined) {
              setUsbcOffline(false);
              setDetectionResult(telem);
              if (onTelemetryUpdate) onTelemetryUpdate(telem);
            }
          }
        } catch (e) {
          console.debug('USB-C telemetry poll error:', e);
          usbcErrorCountRef.current += 1;
          if (usbcErrorCountRef.current >= 5) {
             setErrorMessage('Backend is unreachable. USB-C telemetry stopped.');
             if (usbcTelemetryIntervalRef.current) {
               clearInterval(usbcTelemetryIntervalRef.current);
               usbcTelemetryIntervalRef.current = null;
             }
          }
        }
      };
      poll();
      usbcTelemetryIntervalRef.current = setInterval(poll, 1200);
    } else {
      if (usbcTelemetryIntervalRef.current) {
        clearInterval(usbcTelemetryIntervalRef.current);
        usbcTelemetryIntervalRef.current = null;
      }
    }
    return () => {
      if (usbcTelemetryIntervalRef.current) clearInterval(usbcTelemetryIntervalRef.current);
    };
  }, [isUsbcStreaming, usbcDeviceIndex]);

  // ───────────────────────────────────────────────────────────────────────────
  // Mode 2: Webcam Handlers (detect_webcam.py)
  // ───────────────────────────────────────────────────────────────────────────
  const startWebcam = async () => {
    setErrorMessage('');
    try {
      const constraints = {
        video: selectedDeviceId
          ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 640 }, height: { ideal: 480 } }
          : { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      webcamStreamRef.current = stream;
      if (webcamVideoRef.current) {
        webcamVideoRef.current.srcObject = stream;
        await webcamVideoRef.current.play();
      }
      setIsWebcamActive(true);
      await enumerateCameras();
    } catch (err) {
      console.error('Webcam start error:', err);
      setErrorMessage(`Camera access failed (${err.message}). Ensure camera permissions are granted in browser.`);
    }
  };

  const stopWebcam = () => {
    if (webcamStreamRef.current) {
      webcamStreamRef.current.getTracks().forEach((t) => t.stop());
      webcamStreamRef.current = null;
    }
    if (webcamIntervalRef.current) {
      clearInterval(webcamIntervalRef.current);
      webcamIntervalRef.current = null;
    }
    setIsWebcamActive(false);
  };

  const captureWebcamFrame = async () => {
    if (!webcamVideoRef.current || !webcamCanvasRef.current || !isWebcamActive) return;

    try {
      setIsDetecting(true);
      const video = webcamVideoRef.current;
      const canvas = webcamCanvasRef.current;
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const base64Img = canvas.toDataURL('image/jpeg', 0.85);
      const res = await detectFrame(base64Img, station?.camera_id || 'CAM_VELACHERY_01', confThreshold);
      setDetectionResult(res);
      if (onTelemetryUpdate) onTelemetryUpdate(res);
      setErrorMessage('');
    } catch (err) {
      console.error('Webcam frame detection error:', err);
    } finally {
      setIsDetecting(false);
    }
  };

  useEffect(() => {
    if (isWebcamActive) {
      webcamIntervalRef.current = setInterval(captureWebcamFrame, 1800);
      captureWebcamFrame();
    } else {
      if (webcamIntervalRef.current) clearInterval(webcamIntervalRef.current);
    }
    return () => {
      if (webcamIntervalRef.current) clearInterval(webcamIntervalRef.current);
    };
  }, [isWebcamActive, confThreshold]);

  // ───────────────────────────────────────────────────────────────────────────
  // Mode 3: Image File Detection Handlers (detect_image.py)
  // ───────────────────────────────────────────────────────────────────────────
  const handleImageUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadedImageName(file.name);
    setUploadedImagePreview(URL.createObjectURL(file));

    try {
      setIsDetecting(true);
      setErrorMessage('');
      const res = await uploadMedia(file, station?.camera_id || 'CAM_VELACHERY_01', confThreshold);
      setDetectionResult(res);
      if (onTelemetryUpdate) onTelemetryUpdate(res);
    } catch (err) {
      console.error('Image detection error:', err);
      setErrorMessage('YOLO detection error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setIsDetecting(false);
      if (e.target) e.target.value = '';
    }
  };

  // ───────────────────────────────────────────────────────────────────────────
  // Mode 4: Video Survey Detection Handlers (detect_video.py)
  // ───────────────────────────────────────────────────────────────────────────
  const handleVideoUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadedVideoName(file.name);
    const videoUrl = URL.createObjectURL(file);
    setUploadedVideoUrl(videoUrl);
    setErrorMessage('');
  };

  const analyzeVideoCurrentFrame = async () => {
    if (!videoSurveyPlayerRef.current || !videoSurveyCanvasRef.current) return;
    const video = videoSurveyPlayerRef.current;
    const canvas = videoSurveyCanvasRef.current;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    try {
      setIsDetecting(true);
      const base64Img = canvas.toDataURL('image/jpeg', 0.85);
      const res = await detectFrame(base64Img, station?.camera_id || 'CAM_VELACHERY_01', confThreshold);
      setDetectionResult(res);
      if (onTelemetryUpdate) onTelemetryUpdate(res);
    } catch (err) {
      console.error('Video frame analysis error:', err);
    } finally {
      setIsDetecting(false);
    }
  };

  const toggleVideoSurveyAnalysis = () => {
    if (isVideoAnalyzing) {
      if (videoSurveyIntervalRef.current) clearInterval(videoSurveyIntervalRef.current);
      videoSurveyIntervalRef.current = null;
      setIsVideoAnalyzing(false);
      if (videoSurveyPlayerRef.current) videoSurveyPlayerRef.current.pause();
    } else {
      if (videoSurveyPlayerRef.current) videoSurveyPlayerRef.current.play();
      setIsVideoAnalyzing(true);
      analyzeVideoCurrentFrame();
      videoSurveyIntervalRef.current = setInterval(analyzeVideoCurrentFrame, 1500);
    }
  };

  // Telemetry Calculations matching modelyolo/config.py
  const totalItems = detectionResult?.total_items ?? (station?.debris_count || 6);
  const severity = detectionResult?.severity_label || (totalItems === 0 ? 'CLEAN LAND' : totalItems <= 3 ? 'LOW WASTE' : totalItems <= 8 ? 'MODERATE WASTE' : 'CRITICAL POLLUTION');
  const blockage = detectionResult?.blockage_percentage ?? (station?.blockage_percentage || 78.0);
  const waterCoverage = detectionResult?.water_coverage_percentage ?? (blockage > 0 ? 65.0 : 20.0);
  const classCounts = detectionResult?.class_counts || { 'Plastic / Bottle': 3, 'Polythene / Bag': 2, 'Organic / Debris': 1 };
  const inferenceMs = detectionResult?.inference_time_ms ?? 24.8;

  // Severity color badge resolver
  const getSeverityBadgeClass = (sev) => {
    if (sev === 'CLEAN LAND') return 'badge-low';
    if (sev === 'LOW WASTE') return 'badge-medium';
    if (sev === 'MODERATE WASTE') return 'badge-high';
    return 'badge-critical';
  };

  return (
    <div className="sidebar-content">
      {/* Hidden File Pickers */}
      <input
        type="file"
        ref={imageInputRef}
        onChange={handleImageUpload}
        accept="image/*"
        style={{ display: 'none' }}
      />
      <input
        type="file"
        ref={videoInputRef}
        onChange={handleVideoUpload}
        accept="video/*"
        style={{ display: 'none' }}
      />
      <input
        type="file"
        ref={phoneNativeCameraRef}
        onChange={handleImageUpload}
        accept="image/*"
        capture="environment"
        style={{ display: 'none' }}
      />

      {/* Header Bar */}
      <div style={{ borderBottom: '1px solid var(--border)', paddingBottom: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h2 style={{ fontSize: 15, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 8, color: '#f8fafc' }}>
            <Video style={{ width: 18, height: 18, color: '#38bdf8' }} />
            Land Waste & Drain Camera AI
          </h2>
          <span className="badge badge-critical">STATION 1 (VELACHERY)</span>
        </div>
        <p style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
          <b>Model Engine:</b> YOLO11 Custom Trash (8 Classes) + COCO Object Architecture
        </p>
      </div>

      {/* ── 5 MODES SWITCHER ─────────────────────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', background: 'var(--bg-darkest)', padding: 4, borderRadius: 8, gap: 3 }}>
        <button
          onClick={() => handleModeChange('mobile')}
          className={`tab-button ${activeMode === 'mobile' ? 'active' : ''}`}
          style={{ justifyContent: 'center', padding: '7px 2px', fontSize: 10, fontWeight: 600, borderRadius: 6 }}
          title="Phone IP Webcam / RTSP Stream"
        >
          <Wifi style={{ width: 12, height: 12, marginRight: 3, color: activeMode === 'mobile' ? '#38bdf8' : '#94a3b8' }} />
          IP Cam
        </button>

        <button
          onClick={() => handleModeChange('webcam')}
          className={`tab-button ${activeMode === 'webcam' ? 'active' : ''}`}
          style={{ justifyContent: 'center', padding: '7px 2px', fontSize: 10, fontWeight: 600, borderRadius: 6 }}
          title="Local Webcam via browser getUserMedia"
        >
          <Camera style={{ width: 12, height: 12, marginRight: 3, color: activeMode === 'webcam' ? '#10b981' : '#94a3b8' }} />
          Webcam
        </button>

        <button
          onClick={() => handleModeChange('usbc')}
          className={`tab-button ${activeMode === 'usbc' ? 'active' : ''}`}
          style={{ justifyContent: 'center', padding: '7px 2px', fontSize: 10, fontWeight: 600, borderRadius: 6 }}
          title="USB-C External Camera — backend MJPEG stream with YOLO"
        >
          <Usb style={{ width: 12, height: 12, marginRight: 3, color: activeMode === 'usbc' ? '#a78bfa' : '#94a3b8' }} />
          USB-C
        </button>

        <button
          onClick={() => handleModeChange('image')}
          className={`tab-button ${activeMode === 'image' ? 'active' : ''}`}
          style={{ justifyContent: 'center', padding: '7px 2px', fontSize: 10, fontWeight: 600, borderRadius: 6 }}
          title="Static Image File Detection"
        >
          <ImageIcon style={{ width: 12, height: 12, marginRight: 3, color: activeMode === 'image' ? '#f59e0b' : '#94a3b8' }} />
          Image
        </button>

        <button
          onClick={() => handleModeChange('video')}
          className={`tab-button ${activeMode === 'video' ? 'active' : ''}`}
          style={{ justifyContent: 'center', padding: '7px 2px', fontSize: 10, fontWeight: 600, borderRadius: 6 }}
          title="Video Survey Detection"
        >
          <Video style={{ width: 12, height: 12, marginRight: 3, color: activeMode === 'video' ? '#ec4899' : '#94a3b8' }} />
          Video
        </button>
      </div>

      {/* ═══════════════════════════════════════════════════════════════════════
          MODE 1: PHONE / IP WEBCAM STREAM (detect_ip_webcam.py)
      ═══════════════════════════════════════════════════════════════════════ */}
      {activeMode === 'mobile' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div className="card-box" style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 4 }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: 6 }}>
                <Wifi style={{ width: 14, height: 14, color: isIpStreaming ? '#10b981' : '#38bdf8' }} />
                Phone IP Camera Address
              </span>

              {/* Clean Preset Buttons */}
              <div style={{ display: 'flex', gap: 4 }}>
                <button
                  onClick={() => {
                    const host = ipUrl.split('/')[2]?.split(':')[0] || '192.168.137.1';
                    setIpUrl(`http://${host}:8080/video`);
                  }}
                  className="tab-button"
                  style={{ fontSize: 10, padding: '2px 6px', height: 'auto' }}
                  title="Android IP Webcam default port :8080"
                >
                  :8080 IPWebcam
                </button>
                <button
                  onClick={() => {
                    const host = ipUrl.split('/')[2]?.split(':')[0] || '192.168.137.1';
                    setIpUrl(`http://${host}:4747/video`);
                  }}
                  className="tab-button"
                  style={{ fontSize: 10, padding: '2px 6px', height: 'auto' }}
                  title="DroidCam default port :4747"
                >
                  :4747 DroidCam
                </button>
                <button
                  onClick={() => setIpUrl('http://127.0.0.1:4747/video')}
                  className="tab-button"
                  style={{ fontSize: 10, padding: '2px 6px', height: 'auto' }}
                  title="USB ADB Forwarding Loopback"
                >
                  USB Loopback
                </button>
              </div>
            </div>

            <div style={{ display: 'flex', gap: 6 }}>
              <input
                type="text"
                value={ipUrl}
                onChange={(e) => { setIpUrl(e.target.value); setErrorMessage(''); setStreamOffline(false); }}
                className="input-field"
                placeholder="http://192.168.137.1:8080/video"
                disabled={isIpStreaming}
                style={{ fontSize: 12, fontFamily: 'monospace', flex: 1 }}
              />
              <button
                onClick={handleAutoDiscover}
                disabled={isAutoDiscovering || isIpStreaming}
                className="tab-button"
                style={{
                  fontSize: 11,
                  padding: '4px 10px',
                  height: 'auto',
                  background: 'rgba(56, 189, 248, 0.12)',
                  color: '#38bdf8',
                  borderColor: '#38bdf8',
                  whiteSpace: 'nowrap'
                }}
                title="Scan local subnets for broadcasting phone cameras"
              >
                <Search style={{ width: 12, height: 12, marginRight: 4 }} className={isAutoDiscovering ? 'spin' : ''} />
                {isAutoDiscovering ? 'Scanning...' : 'Auto-Find'}
              </button>
            </div>

            {/* Actions */}
            <div style={{ display: 'flex', gap: 8 }}>
              <button
                onClick={handleIpScanSingleFrame}
                disabled={isDetecting || isIpStreaming}
                className="btn-action btn-blue"
                style={{ flex: 1.2, fontWeight: 700 }}
              >
                <RefreshCw style={{ width: 14, height: 14 }} className={isDetecting ? 'spin' : ''} />
                {isDetecting ? 'Running YOLO...' : 'Scan & Detect'}
              </button>

              {!isIpStreaming ? (
                <button
                  onClick={handleStartIpStream}
                  className="btn-action btn-secondary"
                  style={{ flex: 1, borderColor: '#10b981', color: '#10b981', fontWeight: 600 }}
                >
                  <Play style={{ width: 14, height: 14 }} /> Live Stream
                </button>
              ) : (
                <button
                  onClick={handleStopIpStream}
                  className="btn-action btn-danger"
                  style={{ flex: 1 }}
                >
                  <Square style={{ width: 14, height: 14 }} /> Stop Stream
                </button>
              )}
            </div>

            {/* Direct Phone Snapshot fallback */}
            <div style={{ display: 'flex', gap: 6, paddingTop: 4, borderTop: '1px solid var(--border)' }}>
              <button
                onClick={() => phoneNativeCameraRef.current?.click()}
                disabled={isDetecting}
                className="btn-action btn-secondary"
                style={{ flex: 1, fontSize: 11, padding: '6px 8px' }}
              >
                <Camera style={{ width: 13, height: 13, marginRight: 4, color: '#38bdf8' }} />
                Snap Photo with Phone
              </button>
            </div>
          </div>

          {/* Quick Connection Guidance Card */}
          <div style={{ padding: '8px 12px', background: 'rgba(15, 23, 42, 0.6)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 11, color: '#94a3b8', lineHeight: 1.5 }}>
            <div style={{ fontWeight: 600, color: '#f1f5f9', display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
              <HelpCircle style={{ width: 13, height: 13, color: '#38bdf8' }} />
              How to Connect Phone Camera:
            </div>
            <ol style={{ paddingLeft: 16, margin: 0 }}>
              <li>Open <b>IP Webcam</b> or <b>DroidCam</b> on your phone.</li>
              <li>Tap <b>Start Server</b> and check the IP on your phone screen (e.g. <code style={{ color: '#38bdf8' }}>http://192.168.137.xxx:8080</code>).</li>
              <li>Make sure phone and PC share the <b>same Wi-Fi or Hotspot</b>.</li>
            </ol>
          </div>

          {/* IP Stream Viewport — uses img tag; periodic src refresh prevents first-frame freeze */}
          <div className="camera-frame" style={{ minHeight: 240, position: 'relative' }}>
            {isIpStreaming ? (
              <>
                {/* Key forces a full remount; ipRefreshIntervalRef then busts cache every 80ms */}
                <img
                  key={streamKey}
                  ref={ipImgRef}
                  src={`${API_BASE}/drainwatch/video-feed?stream_url=${encodeURIComponent(ipUrl)}&_t=${streamKey}`}
                  alt="Live IP Feed"
                  style={{
                    width: '100%',
                    height: 'auto',
                    display: 'block',
                    minHeight: 240,
                    background: '#0f172a',
                    objectFit: 'cover',
                  }}
                  onError={() => setStreamOffline(true)}
                  onLoad={() => setStreamOffline(false)}
                />
                <div className="camera-osd">
                  <span style={{ color: streamOffline ? '#ef4444' : '#10b981', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 5 }}>
                    <span style={{ width: 8, height: 8, borderRadius: '50%', background: streamOffline ? '#ef4444' : '#10b981', display: 'inline-block', boxShadow: streamOffline ? 'none' : '0 0 6px #10b981' }} />
                    {streamOffline ? 'STREAM OFFLINE • CHECK PHONE IP' : 'LIVE IP STREAM • YOLO11 AI'}
                  </span>
                  <span>{streamFps > 0 ? `${streamFps} FPS` : '...'} • {inferenceMs} ms</span>
                </div>
              </>
            ) : detectionResult?.annotated_image ? (
              <div>
                <img src={detectionResult.annotated_image} alt="YOLO11 Detection" style={{ width: '100%', display: 'block' }} />
                <div className="camera-osd">
                  <span style={{ color: '#f59e0b', fontWeight: 700 }}>YOLO11 PREDICTION • {severity}</span>
                  <span>{inferenceMs} ms</span>
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '32px 20px' }}>
                <Wifi style={{ width: 40, height: 40, color: '#38bdf8', margin: '0 auto 12px' }} />
                <div style={{ fontSize: 13, fontWeight: 600, color: '#cbd5e1', marginBottom: 6 }}>
                  IP Webcam Stream Ready
                </div>
                <p style={{ fontSize: 11, color: '#94a3b8', maxWidth: 320, margin: '0 auto 14px', lineHeight: 1.5 }}>
                  Enter your phone DroidCam or IP Webcam URL and click <b>Scan & Detect</b>, or start a live <b>MJPEG Stream</b>.
                </p>
                <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
                  <button onClick={handleIpScanSingleFrame} disabled={isDetecting} className="btn-action btn-blue" style={{ width: 'auto' }}>
                    <RefreshCw style={{ width: 14, height: 14 }} className={isDetecting ? 'spin' : ''} />
                    {isDetecting ? 'Running Model...' : 'Scan & Detect'}
                  </button>
                  <button onClick={() => handleModeChange('webcam')} className="btn-action btn-secondary" style={{ width: 'auto' }}>
                    <Camera style={{ width: 14, height: 14 }} /> Use PC Webcam
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════════════
          MODE 3: USB-C EXTERNAL CAMERA
      ═══════════════════════════════════════════════════════════════════════ */}
      {activeMode === 'usbc' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div className="card-box" style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 4 }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: 6 }}>
                <Usb style={{ width: 14, height: 14, color: '#a78bfa' }} />
                USB-C External Camera
              </span>
              <span style={{ fontSize: 10, color: '#64748b', fontStyle: 'italic' }}>Backend MJPEG · YOLO11 Real-time</span>
            </div>

            {/* Device Index Selector */}
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <span style={{ fontSize: 11, color: '#94a3b8', whiteSpace: 'nowrap' }}>Device Index:</span>
              <select
                value={usbcDeviceIndex}
                onChange={(e) => {
                  setUsbcDeviceIndex(parseInt(e.target.value));
                  if (isUsbcStreaming) handleStopUsbcStream();
                }}
                className="input-field"
                style={{ fontSize: 12, flex: 1 }}
                disabled={isUsbcStreaming}
              >
                {[0,1,2,3,4].map(i => (
                  <option key={i} value={i}>Device {i}{i === 0 ? ' (default)' : ''}</option>
                ))}
              </select>
            </div>

            <p style={{ fontSize: 11, color: '#94a3b8', margin: 0, lineHeight: 1.5 }}>
              Connect your camera via <b style={{ color: '#a78bfa' }}>USB-C</b>. The backend reads the device directly and streams live YOLO11-annotated frames. Device 0 is usually the built-in camera; try 1, 2… for external USB-C cameras.
            </p>

            {/* Start / Stop */}
            <div style={{ display: 'flex', gap: 8 }}>
              {!isUsbcStreaming ? (
                <button
                  onClick={handleStartUsbcStream}
                  className="btn-action btn-blue"
                  style={{ flex: 1, background: '#7c3aed', borderColor: '#7c3aed', fontWeight: 700 }}
                >
                  <Play style={{ width: 14, height: 14 }} /> Start USB-C Stream
                </button>
              ) : (
                <button
                  onClick={handleStopUsbcStream}
                  className="btn-action btn-danger"
                  style={{ flex: 1 }}
                >
                  <Square style={{ width: 14, height: 14 }} /> Stop USB-C
                </button>
              )}
            </div>
          </div>

          {/* USB-C Viewport */}
          <div className="camera-frame" style={{ minHeight: 240, position: 'relative' }}>
            {isUsbcStreaming ? (
              <>
                <img
                  key={usbcStreamKey}
                  src={`${usbcFeedUrl(usbcDeviceIndex)}&_t=${usbcStreamKey}`}
                  alt="USB-C Live Feed"
                  style={{
                    width: '100%',
                    height: 'auto',
                    display: 'block',
                    minHeight: 240,
                    background: '#0f172a',
                    objectFit: 'cover',
                  }}
                  onError={() => setUsbcOffline(true)}
                  onLoad={() => setUsbcOffline(false)}
                />
                <div className="camera-osd">
                  <span style={{ color: usbcOffline ? '#ef4444' : '#a78bfa', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 5 }}>
                    <span style={{ width: 8, height: 8, borderRadius: '50%', background: usbcOffline ? '#ef4444' : '#a78bfa', display: 'inline-block', boxShadow: usbcOffline ? 'none' : '0 0 6px #a78bfa' }} />
                    {usbcOffline ? `DEVICE ${usbcDeviceIndex} OFFLINE` : `USB-C LIVE • DEVICE ${usbcDeviceIndex} • YOLO11`}
                  </span>
                  <span>{usbcFps > 0 ? `${usbcFps} FPS` : '...'} • {inferenceMs} ms</span>
                </div>
              </>
            ) : (
              <div style={{ textAlign: 'center', padding: '32px 20px' }}>
                <Usb style={{ width: 40, height: 40, color: '#a78bfa', margin: '0 auto 12px' }} />
                <div style={{ fontSize: 13, fontWeight: 600, color: '#cbd5e1', marginBottom: 6 }}>
                  USB-C Camera Ready
                </div>
                <p style={{ fontSize: 11, color: '#94a3b8', maxWidth: 320, margin: '0 auto 14px', lineHeight: 1.5 }}>
                  Plug in your USB-C camera, select the device index above, and click <b>Start USB-C Stream</b>.
                </p>
                <button
                  onClick={handleStartUsbcStream}
                  className="btn-action btn-blue"
                  style={{ width: 'auto', margin: '0 auto', background: '#7c3aed', borderColor: '#7c3aed' }}
                >
                  <Play style={{ width: 14, height: 14 }} /> Start USB-C Stream
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════════════
          MODE 2: LAPTOP / USB WEBCAM (detect_webcam.py)
      ═══════════════════════════════════════════════════════════════════════ */}
      {activeMode === 'webcam' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div className="card-box" style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: 6 }}>
                <Camera style={{ width: 14, height: 14, color: '#10b981' }} />
                Device Webcam Selection
              </span>
              <button
                onClick={enumerateCameras}
                style={{ background: 'none', border: 'none', color: '#38bdf8', cursor: 'pointer', fontSize: 11, display: 'flex', alignItems: 'center', gap: 3 }}
              >
                <RefreshCw style={{ width: 11, height: 11 }} /> Refresh
              </button>
            </div>

            <select
              value={selectedDeviceId}
              onChange={(e) => {
                setSelectedDeviceId(e.target.value);
                if (isWebcamActive) stopWebcam();
              }}
              className="input-field"
              style={{ fontSize: 12 }}
            >
              {videoDevices.length > 0 ? (
                videoDevices.map((d, i) => (
                  <option key={d.deviceId || i} value={d.deviceId}>
                    {d.label || `Camera Device ${i + 1}`}
                  </option>
                ))
              ) : (
                <option value="">Default Built-in / USB Camera</option>
              )}
            </select>

            <div style={{ display: 'flex', gap: 8 }}>
              {!isWebcamActive ? (
                <button
                  onClick={startWebcam}
                  className="btn-action btn-blue"
                  style={{ flex: 1, background: '#10b981', borderColor: '#10b981', fontWeight: 700 }}
                >
                  <Play style={{ width: 14, height: 14 }} /> Start Webcam Detection
                </button>
              ) : (
                <>
                  <button
                    onClick={captureWebcamFrame}
                    disabled={isDetecting}
                    className="btn-action btn-secondary"
                    style={{ flex: 1 }}
                  >
                    <RefreshCw style={{ width: 14, height: 14 }} className={isDetecting ? 'spin' : ''} />
                    {isDetecting ? 'Detecting...' : 'Force Re-scan'}
                  </button>
                  <button
                    onClick={stopWebcam}
                    className="btn-action btn-danger"
                    style={{ width: 'auto' }}
                  >
                    <Square style={{ width: 14, height: 14 }} /> Stop
                  </button>
                </>
              )}
            </div>
          </div>

          {/* Webcam Viewport */}
          <div className="camera-frame" style={{ minHeight: 240, position: 'relative' }}>
            <video ref={webcamVideoRef} playsInline muted style={{ display: isWebcamActive ? 'block' : 'none', width: '100%' }} />
            <canvas ref={webcamCanvasRef} style={{ display: 'none' }} />

            {isWebcamActive && (
              <div className="camera-osd">
                <span style={{ color: '#10b981', fontWeight: 700 }}>● LIVE WEBCAM • YOLO11 REAL-TIME</span>
                <span>{inferenceMs} ms • {severity}</span>
              </div>
            )}

            {!isWebcamActive && detectionResult?.annotated_image && (
              <div>
                <img src={detectionResult.annotated_image} alt="YOLO11 Detection" style={{ width: '100%', display: 'block' }} />
                <div className="camera-osd">
                  <span style={{ color: '#f59e0b', fontWeight: 700 }}>YOLO11 PREDICTION • {severity}</span>
                  <span>{inferenceMs} ms</span>
                </div>
              </div>
            )}

            {!isWebcamActive && !detectionResult?.annotated_image && (
              <div style={{ textAlign: 'center', padding: '32px 20px' }}>
                <Camera style={{ width: 40, height: 40, color: '#10b981', margin: '0 auto 12px' }} />
                <div style={{ fontSize: 13, fontWeight: 600, color: '#cbd5e1', marginBottom: 6 }}>
                  Laptop / USB Camera Ready
                </div>
                <p style={{ fontSize: 11, color: '#94a3b8', maxWidth: 320, margin: '0 auto 14px', lineHeight: 1.5 }}>
                  Click <b>Start Webcam</b> to begin continuous real-time YOLO11 waste & debris detection.
                </p>
                <button
                  onClick={startWebcam}
                  className="btn-action btn-blue"
                  style={{ width: 'auto', margin: '0 auto', background: '#10b981', borderColor: '#10b981' }}
                >
                  <Play style={{ width: 14, height: 14 }} /> Start Webcam Detection
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════════════
          MODE 3: STATIC IMAGE FILE DETECTION (detect_image.py)
      ═══════════════════════════════════════════════════════════════════════ */}
      {activeMode === 'image' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div className="card-box" style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: 6 }}>
              <ImageIcon style={{ width: 14, height: 14, color: '#f59e0b' }} />
              Static Image Detection (detect_image.py)
            </span>

            <div
              onClick={() => imageInputRef.current?.click()}
              style={{
                border: '2px dashed rgba(255, 255, 255, 0.15)',
                borderRadius: 8,
                padding: '24px 16px',
                textAlign: 'center',
                cursor: 'pointer',
                background: 'rgba(255, 255, 255, 0.02)',
                transition: 'all 0.2s ease',
              }}
            >
              <Upload style={{ width: 28, height: 28, color: '#f59e0b', margin: '0 auto 8px' }} />
              <div style={{ fontSize: 12, fontWeight: 600, color: '#f1f5f9' }}>
                {uploadedImageName ? `Selected: ${uploadedImageName}` : 'Click or Drop Image File to Run YOLO11'}
              </div>
              <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
                Supports JPG, PNG, WEBP high-resolution drain inspection photos
              </div>
            </div>
          </div>

          {/* Image Viewport */}
          <div className="camera-frame" style={{ minHeight: 240, position: 'relative' }}>
            {detectionResult?.annotated_image ? (
              <div>
                <img src={detectionResult.annotated_image} alt="YOLO11 Detection" style={{ width: '100%', display: 'block' }} />
                <div className="camera-osd">
                  <span style={{ color: '#f59e0b', fontWeight: 700 }}>YOLO11 PREDICTION • {severity}</span>
                  <span>{inferenceMs} ms • {totalItems} Items</span>
                </div>
              </div>
            ) : uploadedImagePreview ? (
              <img src={uploadedImagePreview} alt="Uploaded preview" style={{ width: '100%', display: 'block' }} />
            ) : (
              <div style={{ textAlign: 'center', padding: '32px 20px' }}>
                <ImageIcon style={{ width: 40, height: 40, color: '#f59e0b', margin: '0 auto 12px' }} />
                <div style={{ fontSize: 13, fontWeight: 600, color: '#cbd5e1', marginBottom: 6 }}>
                  No Image Selected
                </div>
                <p style={{ fontSize: 11, color: '#94a3b8', maxWidth: 300, margin: '0 auto 12px' }}>
                  Select an image of a stormwater drain, roadside canal, or land litter to run instant YOLO11 detection.
                </p>
                <button
                  onClick={() => imageInputRef.current?.click()}
                  className="btn-action btn-blue"
                  style={{ width: 'auto', margin: '0 auto', background: '#f59e0b', borderColor: '#f59e0b' }}
                >
                  <Upload style={{ width: 14, height: 14 }} /> Choose Image File
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════════════
          MODE 4: VIDEO SURVEY DETECTION (detect_video.py)
      ═══════════════════════════════════════════════════════════════════════ */}
      {activeMode === 'video' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div className="card-box" style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: 6 }}>
              <Video style={{ width: 14, height: 14, color: '#ec4899' }} />
              Video Survey Detection (detect_video.py)
            </span>

            <div style={{ display: 'flex', gap: 8 }}>
              <button
                onClick={() => videoInputRef.current?.click()}
                className="btn-action btn-secondary"
                style={{ flex: 1, borderColor: '#ec4899', color: '#ec4899' }}
              >
                <Upload style={{ width: 14, height: 14 }} />
                {uploadedVideoName ? `Video: ${uploadedVideoName}` : 'Upload Survey Video'}
              </button>

              {uploadedVideoUrl && (
                <button
                  onClick={toggleVideoSurveyAnalysis}
                  className={`btn-action ${isVideoAnalyzing ? 'btn-danger' : 'btn-blue'}`}
                  style={{ flex: 1 }}
                >
                  {isVideoAnalyzing ? (
                    <>
                      <Square style={{ width: 14, height: 14 }} /> Pause Analysis
                    </>
                  ) : (
                    <>
                      <Play style={{ width: 14, height: 14 }} /> Run Video YOLO
                    </>
                  )}
                </button>
              )}
            </div>
          </div>

          {/* Video Viewport */}
          <div className="camera-frame" style={{ minHeight: 240, position: 'relative' }}>
            <canvas ref={videoSurveyCanvasRef} style={{ display: 'none' }} />
            {uploadedVideoUrl ? (
              <>
                <video
                  ref={videoSurveyPlayerRef}
                  src={uploadedVideoUrl}
                  controls
                  playsInline
                  style={{ width: '100%', height: 'auto', display: 'block' }}
                />
                {isVideoAnalyzing && (
                  <div className="camera-osd">
                    <span style={{ color: '#ec4899', fontWeight: 700 }}>● VIDEO SURVEY YOLO ANALYSIS</span>
                    <span>{inferenceMs} ms • {severity}</span>
                  </div>
                )}
              </>
            ) : (
              <div style={{ textAlign: 'center', padding: '32px 20px' }}>
                <Video style={{ width: 40, height: 40, color: '#ec4899', margin: '0 auto 12px' }} />
                <div style={{ fontSize: 13, fontWeight: 600, color: '#cbd5e1', marginBottom: 6 }}>
                  Drone & Survey Video Ready
                </div>
                <p style={{ fontSize: 11, color: '#94a3b8', maxWidth: 300, margin: '0 auto 12px' }}>
                  Upload drone survey or CCTV footage (MP4, WEBM) for sequential frame-by-frame debris classification.
                </p>
                <button
                  onClick={() => videoInputRef.current?.click()}
                  className="btn-action btn-blue"
                  style={{ width: 'auto', margin: '0 auto', background: '#ec4899', borderColor: '#ec4899' }}
                >
                  <Upload style={{ width: 14, height: 14 }} /> Select Video File
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Confidence Threshold Slider (YOLO Parameter from config.py) ─────── */}
      <div className="card-box" style={{ padding: '10px 14px', display: 'flex', flexDirection: 'column', gap: 6 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 11 }}>
          <span style={{ color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 5 }}>
            <Sliders style={{ width: 12, height: 12, color: '#38bdf8' }} />
            Model Confidence Threshold (YOLO11)
          </span>
          <span style={{ fontWeight: 700, color: '#38bdf8', fontFamily: 'monospace' }}>
            {(confThreshold * 100).toFixed(0)}%
          </span>
        </div>
        <input
          type="range"
          min="0.05"
          max="0.80"
          step="0.05"
          value={confThreshold}
          onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
          style={{ width: '100%', accentColor: '#38bdf8', cursor: 'pointer' }}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#64748b' }}>
          <span>5% (High Sensitivity)</span>
          <span>15% (Default)</span>
          <span>80% (Strict)</span>
        </div>
      </div>

      {/* ── Error Banner ────────────────────────────────────────────────────── */}
      {errorMessage && (
        <div style={{ padding: '10px 14px', background: 'rgba(239, 68, 68, 0.12)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: 8, fontSize: 11, color: '#fca5a5', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 10 }}>
          <div style={{ lineHeight: 1.5, flex: 1 }}>
            <b>Notice:</b> {errorMessage}
          </div>
          <button
            onClick={() => setErrorMessage('')}
            style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', fontSize: 11, textDecoration: 'underline', padding: 0 }}
          >
            Dismiss
          </button>
        </div>
      )}

      {/* ── Real Model Waste & Hydrological Telemetry HUD ──────────────────── */}
      <div className="card-box" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border)', paddingBottom: 8 }}>
          <span style={{ fontSize: 12, fontWeight: 700, color: '#f1f5f9', display: 'flex', alignItems: 'center', gap: 6 }}>
            <Sparkles style={{ width: 14, height: 14, color: '#f59e0b' }} />
            Land Waste & Drain Status
          </span>
          <span className={`badge ${getSeverityBadgeClass(severity)}`}>
            {severity}
          </span>
        </div>

        {/* Blockage Gauge */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 4 }}>
            <span style={{ color: '#94a3b8' }}>Inlet Debris Blockage</span>
            <span style={{ fontWeight: 700, color: blockage >= 70 ? '#ef4444' : blockage >= 40 ? '#f59e0b' : '#10b981', fontFamily: 'monospace' }}>
              {blockage}%
            </span>
          </div>
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{
                width: `${blockage}%`,
                background: blockage >= 70 ? '#ef4444' : blockage >= 40 ? '#f59e0b' : '#10b981',
                transition: 'width 0.4s ease',
              }}
            />
          </div>
        </div>

        {/* Water Level Gauge */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 4 }}>
            <span style={{ color: '#94a3b8' }}>Surface Water Level</span>
            <span style={{ fontWeight: 700, color: '#06b6d4', fontFamily: 'monospace' }}>
              {waterCoverage}%
            </span>
          </div>
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{
                width: `${waterCoverage}%`,
                background: '#06b6d4',
                transition: 'width 0.4s ease',
              }}
            />
          </div>
        </div>

        {/* 8 Waste Classes Identified */}
        <div>
          <span className="card-title" style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>Debris Classified ({totalItems} Total Objects)</span>
            <span style={{ color: '#38bdf8', fontSize: 10 }}>{inferenceMs} ms</span>
          </span>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {Object.keys(classCounts).length > 0 ? (
              Object.entries(classCounts).map(([cls, cnt]) => (
                <span
                  key={cls}
                  style={{
                    padding: '3px 8px',
                    background: 'var(--bg-elevated)',
                    borderRadius: 4,
                    fontSize: 11,
                    border: '1px solid var(--border)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 5
                  }}
                >
                  <span
                    style={{
                      width: 7,
                      height: 7,
                      borderRadius: '50%',
                      background: TRASH_CLASS_COLORS[cls] || '#38bdf8',
                      display: 'inline-block'
                    }}
                  />
                  {cls} <b style={{ color: TRASH_CLASS_COLORS[cls] || '#f59e0b' }}>×{cnt}</b>
                </span>
              ))
            ) : (
              <span style={{ fontSize: 11, color: '#64748b', fontStyle: 'italic' }}>No debris detected on target ground</span>
            )}
          </div>
        </div>

        {/* Dynamic Hydrological Route Impact */}
        <div style={{ padding: '8px 10px', background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.25)', borderRadius: 6, fontSize: 11, color: '#cbd5e1', lineHeight: 1.4 }}>
          <b>⚡ Dynamic Safe Routing:</b> Debris count feeds an additive <b>+{((blockage / 100.0) * 0.35).toFixed(2)} risk delta</b> to road segment <code style={{ color: '#38bdf8' }}>seg_tnagar_velachery</code>, automatically recalibrating emergency routes across Chennai.
        </div>
      </div>
    </div>
  );
}
