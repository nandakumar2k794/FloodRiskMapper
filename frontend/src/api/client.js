import axios from 'axios'

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000/api/v1',
  timeout: 15000,
})

// ── Risk Map & Layers ─────────────────────────────────────────────────────────
export const fetchRiskMap = (params = {}) =>
  api.get('/risk-map', { params }).then(r => r.data)

export const fetchLayers = () =>
  api.get('/layers').then(r => r.data)

export const fetchRiskGrid = (limit = 300) =>
  api.get('/risk-map/grid', { params: { limit } }).then(r => r.data)

export const fetchStats = () =>
  api.get('/stats').then(r => r.data)

// ── Routing ───────────────────────────────────────────────────────────────────
export const fetchRoute = (body) =>
  api.post('/route', body).then(r => r.data)

// ── Alerts ────────────────────────────────────────────────────────────────────
export const fetchAlerts = (params = {}) =>
  api.get('/alerts', { params }).then(r => r.data)

// ── DrainWatch ────────────────────────────────────────────────────────────────
export const fetchCctvStations = () =>
  api.get('/drainwatch/stations').then(r => r.data)

export const fetchDrainStatus = (params = {}) =>
  api.get('/drain-status', { params }).then(r => r.data)

export const detectFrame = (imageBase64, cameraId = 'CAM_VELACHERY_01', confThreshold = 0.15) =>
  api.post('/drainwatch/detect-frame', {
    image_base64: imageBase64,
    camera_id: cameraId,
    conf_threshold: confThreshold,
  }).then(r => r.data)

export const detectPhoneStream = (streamUrl, cameraId = 'CAM_VELACHERY_01', confThreshold = 0.15) =>
  api.post('/drainwatch/phone-stream', {
    stream_url: streamUrl,
    camera_id: cameraId,
    conf_threshold: confThreshold,
  }).then(r => r.data)

export const fetchPhoneTelemetry = () =>
  api.get('/drainwatch/phone-telemetry').then(r => r.data)

export const stopPhoneStream = () =>
  api.post('/drainwatch/stop-stream').then(r => r.data)

export const discoverPhoneCamera = () =>
  api.get('/drainwatch/discover-phone').then(r => r.data)

export const uploadMedia = (file, cameraId = 'CAM_VELACHERY_01', confThreshold = 0.15) => {
  const formData = new FormData()
  formData.append('file', file)
  return api.post(`/drainwatch/upload-media?camera_id=${cameraId}&conf_threshold=${confThreshold}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data)
}

// ── USB-C External Camera ──────────────────────────────────────────────────────
export const fetchUsbcTelemetry = () =>
  api.get('/drainwatch/usbc-telemetry').then(r => r.data)

export const stopUsbcStreamApi = () =>
  api.post('/drainwatch/usbc-stop').then(r => r.data)

// ── Rainfall ──────────────────────────────────────────────────────────────────
export const fetchRainfall = () =>
  api.get('/rainfall').then(r => r.data)

export const fetchRainfallHistory = (stationId, hours = 24) =>
  api.get('/rainfall/history', { params: { station_id: stationId, hours } }).then(r => r.data)

// ── MongoDB Critical Paths ────────────────────────────────────────────────────
export const fetchCriticalPaths = (params = {}) =>
  api.get('/critical-paths', { params }).then(r => r.data)

export const fetchCriticalStats = () =>
  api.get('/critical-paths/stats').then(r => r.data)

export const syncCriticalPaths = (rainfall = 140.0) =>
  api.post('/critical-paths/sync', null, { params: { rainfall } }).then(r => r.data)

export default api

