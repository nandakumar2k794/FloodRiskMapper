"""
System health and diagnostics utility for Chennai Flood Access Risk Mapper.
"""
import sys
import platform
import os
from pathlib import Path
from typing import Dict, Any, List
import importlib.metadata

from config.constants import PROJECT_NAME, PROJECT_VERSION


CORE_PACKAGES = [
    {"name": "fastapi", "category": "API / Backend", "required": True},
    {"name": "streamlit", "category": "Frontend / Dashboard", "required": True},
    {"name": "torch", "category": "Deep Learning / CV", "required": True},
    {"name": "torchvision", "category": "Deep Learning / CV", "required": False},
    {"name": "ultralytics", "category": "DrainWatch YOLO", "required": True},
    {"name": "geopandas", "category": "Spatial / GIS", "required": True},
    {"name": "rasterio", "category": "Raster / DEM", "required": True},
    {"name": "osmnx", "category": "Roads / Graph", "required": True},
    {"name": "networkx", "category": "Routing / Graph", "required": True},
    {"name": "shapely", "category": "Spatial Geometry", "required": True},
    {"name": "scikit-learn", "category": "Machine Learning", "required": True},
    {"name": "xgboost", "category": "Flood Risk Model", "required": True},
    {"name": "pandas", "category": "Data Processing", "required": True},
    {"name": "numpy", "category": "Numerical", "required": True},
    {"name": "pydantic", "category": "Configuration", "required": True},
    {"name": "requests", "category": "External APIs", "required": True},
    {"name": "pytest", "category": "Testing", "required": True},
]


def get_python_info() -> Dict[str, Any]:
    """Retrieve Python runtime details."""
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "compiler": platform.python_compiler(),
        "architecture": platform.architecture()[0],
        "system": platform.system(),
        "release": platform.release(),
        "executable": sys.executable,
    }


def get_gpu_info() -> Dict[str, Any]:
    """Inspect GPU / CUDA availability via PyTorch."""
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        device_name = torch.cuda.get_device_name(0) if cuda_available and device_count > 0 else "N/A (CPU Mode)"

        return {
            "torch_version": torch.__version__,
            "cuda_available": cuda_available,
            "device_count": device_count,
            "device_name": device_name,
            "default_device": "cuda:0" if cuda_available else "cpu",
        }
    except Exception as exc:
        return {
            "torch_version": "Not installed",
            "cuda_available": False,
            "device_count": 0,
            "device_name": f"Error: {exc}",
            "default_device": "cpu",
        }


def get_package_statuses() -> List[Dict[str, Any]]:
    """Check installed status and versions of key dependencies."""
    results = []
    for pkg in CORE_PACKAGES:
        pkg_name = pkg["name"]
        status = {"name": pkg_name, "category": pkg["category"], "required": pkg["required"]}
        try:
            ver = importlib.metadata.version(pkg_name)
            status["installed"] = True
            status["version"] = ver
            status["status"] = "OK"
        except importlib.metadata.PackageNotFoundError:
            status["installed"] = False
            status["version"] = "Not Found"
            status["status"] = "Missing"
        results.append(status)
    return results


def get_directory_health() -> Dict[str, bool]:
    """Check existence of all essential project directories."""
    root = Path(__file__).resolve().parent.parent
    check_dirs = [
        "app",
        "cv",
        "ml",
        "gis",
        "routing",
        "data",
        "models",
        "scripts",
        "tests",
        "config",
        "notebooks",
        "logs",
    ]
    return {d: (root / d).is_dir() for d in check_dirs}


def get_system_summary() -> Dict[str, Any]:
    """Aggregate complete system health diagnostics."""
    return {
        "project_name": PROJECT_NAME,
        "project_version": PROJECT_VERSION,
        "python": get_python_info(),
        "gpu": get_gpu_info(),
        "packages": get_package_statuses(),
        "directories": get_directory_health(),
    }
