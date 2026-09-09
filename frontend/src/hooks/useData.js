import { useState, useEffect, useRef, useCallback } from 'react'
import { fetchRiskMap, fetchStats, fetchAlerts, fetchDrainStatus, fetchRainfall, fetchRoute } from '../api/client'

// ── Generic polling hook ──────────────────────────────────────────────────────
export function usePolling(fetcher, interval = 60000, deps = []) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetch = useCallback(async () => {
    try {
      const result = await fetcher()
      setData(result)
      setError(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, deps)

  useEffect(() => {
    fetch()
    const id = setInterval(fetch, interval)
    return () => clearInterval(id)
  }, [fetch, interval])

  return { data, loading, error, refetch: fetch }
}

// ── Risk Map ──────────────────────────────────────────────────────────────────
export function useRiskMap(filters = {}) {
  return usePolling(() => fetchRiskMap(filters), 60000, [JSON.stringify(filters)])
}

// ── Stats ─────────────────────────────────────────────────────────────────────
export function useStats() {
  return usePolling(fetchStats, 30000)
}

// ── Alerts ────────────────────────────────────────────────────────────────────
export function useAlerts() {
  return usePolling(fetchAlerts, 30000)
}

// ── Drain Status ──────────────────────────────────────────────────────────────
export function useDrainStatus() {
  return usePolling(fetchDrainStatus, 45000)
}

// ── Rainfall ──────────────────────────────────────────────────────────────────
export function useRainfall() {
  return usePolling(fetchRainfall, 60000)
}

// ── WebSocket Live Feed ───────────────────────────────────────────────────────
export function useWebSocket(onMessage) {
  const wsRef = useRef(null)
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    const url = `ws://127.0.0.1:8000/ws/live`

    const connect = () => {
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen    = () => setConnected(true)
      ws.onclose   = () => { setConnected(false); setTimeout(connect, 5000) }
      ws.onerror   = () => ws.close()
      ws.onmessage = (e) => {
        try { onMessage(JSON.parse(e.data)) } catch {}
      }
    }

    connect()
    return () => wsRef.current?.close()
  }, [])

  return connected
}

// ── Routing ───────────────────────────────────────────────────────────────────
export function useRoute() {
  const [routes, setRoutes] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const findRoute = useCallback(async (body) => {
    setLoading(true)
    setError(null)
    try {
      const result = await fetchRoute(body)
      setRoutes(result)
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  return { routes, loading, error, findRoute }
}
