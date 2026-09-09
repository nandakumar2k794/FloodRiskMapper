/**
 * Risk color utilities matching backend constants.
 */

export const RISK_COLORS = {
  LOW:      '#22c55e',
  MEDIUM:   '#f59e0b',
  HIGH:     '#f97316',
  CRITICAL: '#ef4444',
}

export const DRAIN_CONDITION_LABELS = {
  drain_clear:             '✅ Clear',
  drain_blocked_garbage:   '🗑️ Blocked – Garbage',
  drain_blocked_silt:      '🪨 Blocked – Silt',
  drain_overflowing:       '🌊 Overflowing',
  standing_water_minor:    '💧 Standing Water',
  standing_water_major:    '🌧️ Major Standing Water',
  road_waterlogged:        '🚧 Road Waterlogged',
}

export const DRAIN_CONDITION_COLORS = {
  drain_clear:             '#22c55e',
  drain_blocked_garbage:   '#f97316',
  drain_blocked_silt:      '#f59e0b',
  drain_overflowing:       '#ef4444',
  standing_water_minor:    '#06b6d4',
  standing_water_major:    '#3b82f6',
  road_waterlogged:        '#ef4444',
}

export const riskColor = (score) => {
  if (score >= 0.75) return RISK_COLORS.CRITICAL
  if (score >= 0.50) return RISK_COLORS.HIGH
  if (score >= 0.25) return RISK_COLORS.MEDIUM
  return RISK_COLORS.LOW
}

export const riskClass = (score) => {
  if (score >= 0.75) return 'CRITICAL'
  if (score >= 0.50) return 'HIGH'
  if (score >= 0.25) return 'MEDIUM'
  return 'LOW'
}

export const formatScore = (s) => (s * 100).toFixed(0) + '%'
export const formatDepth = (d) => d != null ? `${d.toFixed(0)} cm` : '—'
export const formatDist  = (m) => m >= 1000 ? `${(m/1000).toFixed(1)} km` : `${m.toFixed(0)} m`

export const ROUTE_COLORS = ['#3b82f6', '#a855f7', '#f59e0b']
