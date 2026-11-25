#  Flood Access Risk Mapper with DrainWatch AI

> Operational road-level accessibility mapping and CCTV-based visual drainage intelligence for urban flood resilience in Chennai.

## Overview

Chennai experiences intense monsoon rainfall and urban flooding. While macro-scale weather forecasting exists, city operations and citizens need **hyper-local road segment accessibility intelligence**:
1. **Which specific roads will become impassable?**
2. **What are the safest alternative routes for emergency services, buses, and commuters?**
3. **Where are storm drains physically blocked or overflowing right now?**

**DrainWatch AI** is the computer vision intelligence layer embedded into this system. It monitors municipal CCTV and community camera feeds to detect:
- Garbage and debris accumulation in drains
- Silt/mud sedimentation
- Standing water and road waterlogging
- Drain overflows

These visual observations dynamically adjust physical road accessibility weights in the routing and risk engine.

---

## Directory Structure

```text
smartathon/
├── app/                  # Streamlit dashboard & health interface
│   ├── __init__.py
│   └── main.py           # Streamlit application entry point
├── config/               # Configuration, constants & logging
│   ├── __init__.py
│   ├── constants.py      # Thresholds, bounds, weights, CRS
│   ├── settings.py       # Pydantic environment settings
│   └── logging_config.py # Structured logging setup
├── cv/                   # DrainWatch AI computer vision module
│   ├── __init__.py
│   ├── detector.py       # YOLOv8 drain & water detection interface
│   └── drainwatch.py     # CCTV feed processor & risk aggregator
├── data/                 # Spatial, rainfall, and image datasets (git-ignored)
│   ├── raw/              # Raw DEM, OSM, rainfall, video inputs
│   └── processed/        # Processed graphs, raster HAND, features
├── gis/                  # Geographic Information Systems module
│   ├── __init__.py
│   ├── dem.py            # Digital Elevation Model & slope processor
│   ├── hand.py           # Height Above Nearest Drainage calculator
│   └── roads.py          # OSM road network extractor & graph builder
├── ml/                   # Machine learning & flood modeling
│   ├── __init__.py
│   ├── rainfall.py       # IMD / OpenWeather rainfall ingestion
│   ├── features.py       # Road segment feature engineering
│   └── flood_model.py    # Road flood risk scoring engine
├── models/               # Serialized model weights (pt, pkl)
├── notebooks/            # Exploratory research & analysis notebooks
├── routing/              # Inundation-aware alternative routing
│   ├── __init__.py
│   ├── graph.py          # NetworkX weighted road graph interface
│   └── router.py         # Risk-weighted Dijkstra & K-shortest paths
├── scripts/              # Data preprocessing and utility scripts
├── tests/                # Pytest unit & integration test suite
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_health.py
│   └── test_placeholders.py
├── .env.example          # Environment variable template
├── .gitignore            # Git exclusion rules
├── requirements.txt      # Project Python dependencies
└── README.md             # Project documentation
```

---

## Getting Started (Windows / PowerShell)

### 1. Environment Setup

```powershell
# Clone or navigate to the repository
cd d:\smartathon

# Ensure Python 3.10+ is available
python --version

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the `.env.example` template:
```powershell
Copy-Item .env.example .env
```

### 3. Run the Streamlit Dashboard

```powershell
streamlit run app/main.py
```
Open your browser at `http://localhost:8501` to view the System Health & Operational Status dashboard.

### 4. Run Automated Tests

```powershell
python -m pytest tests/ -v
```

---


