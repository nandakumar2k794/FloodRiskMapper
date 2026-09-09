"""
Project-wide constants for Chennai Flood Access Risk Mapper.
"""

PROJECT_NAME = "Chennai Flood Access Risk Mapper with DrainWatch AI"
PROJECT_VERSION = "0.1.0-alpha"

# ─── Chennai Geographic Boundary ──────────────────────────────────────────────
CHENNAI_BBOX = {
    "north": 13.2500,
    "south": 12.8500,
    "east": 80.3300,
    "west": 80.1500,
}

# Centre point for map initialisation
CHENNAI_CENTER = (13.0827, 80.2707)

# ─── Coordinate Reference Systems ─────────────────────────────────────────────
CRS_WGS84 = "EPSG:4326"        # Geographic – storage & display
CRS_UTM44N = "EPSG:32644"      # Projected metric – GIS analysis

# ─── Risk Scoring Thresholds ──────────────────────────────────────────────────
RISK_CLASSES = {
    "LOW":      (0.00, 0.25),
    "MEDIUM":   (0.25, 0.50),
    "HIGH":     (0.50, 0.75),
    "CRITICAL": (0.75, 1.00),
}

RISK_COLORS = {
    "LOW":      "#22c55e",   # green
    "MEDIUM":   "#f59e0b",   # amber
    "HIGH":     "#f97316",   # orange
    "CRITICAL": "#ef4444",   # red
}

# ─── Feature Weights for Composite Risk Score ──────────────────────────────────
RISK_WEIGHTS = {
    "rainfall_score":     0.30,   # 24-h cumulative rainfall
    "topographic_score":  0.25,   # HAND + slope + TWI
    "historical_score":   0.20,   # historical flood frequency
    "drainwatch_score":   0.15,   # CV-derived drain condition
    "proximity_score":    0.10,   # distance to river/lake/drain
}

assert abs(sum(RISK_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1"

# ─── Routing ───────────────────────────────────────────────────────────────────
ROUTE_RISK_ALPHA = 5.0          # risk amplification in edge weight
ROUTE_INACCESSIBLE_PENALTY = 1_000_000
ROUTE_K_ALTERNATIVES = 3

# Depth (cm) above which a road is inaccessible by vehicle type
DEPTH_THRESHOLDS_CM = {
    "car":          25,
    "suv":          40,
    "bus":          50,
    "motorcycle":   15,
    "pedestrian":   10,
}

# ─── Rainfall Accumulation Thresholds (IMD scale) ─────────────────────────────
RAINFALL_THRESHOLDS_MM = {
    "light":     2.5,
    "moderate":  15.6,
    "heavy":     64.5,
    "very_heavy":115.6,
    "extremely_heavy": 204.5,
}

# ─── DrainWatch AI ────────────────────────────────────────────────────────────
DRAINWATCH_CLASSES = [
    "drain_clear",
    "drain_blocked_garbage",
    "drain_blocked_silt",
    "drain_overflowing",
    "standing_water_minor",
    "standing_water_major",
    "road_waterlogged",
]

# Scores assigned to each class (0=safe, 1=critical)
DRAINWATCH_CLASS_SCORES = {
    "drain_clear":            0.0,
    "drain_blocked_garbage":  0.7,
    "drain_blocked_silt":     0.5,
    "drain_overflowing":      0.9,
    "standing_water_minor":   0.4,
    "standing_water_major":   0.8,
    "road_waterlogged":       1.0,
}

DRAINWATCH_WINDOW_FRAMES = 30   # frames per observation window
DRAINWATCH_FPS = 1              # frames sampled per second (CPU mode)

# ─── HAND / DEM Analysis ──────────────────────────────────────────────────────
HAND_FLOOD_THRESHOLD_M = 3.0    # HAND < this → high flood susceptibility
TWI_HIGH_THRESHOLD = 8.0        # TWI > this → wetness hotspot

# ─── Alerts ───────────────────────────────────────────────────────────────────
ALERT_TYPES = ["ROAD_BLOCKED", "DRAIN_OVERFLOW", "HIGH_RISK", "WATERLOGGED"]
ALERT_SEVERITIES = ["INFO", "WARNING", "HIGH", "CRITICAL"]
