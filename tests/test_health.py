"""
Unit tests for system diagnostics and health checking logic.
"""
import pytest
from app.health import (
    get_python_info,
    get_gpu_info,
    get_package_statuses,
    get_directory_health,
    get_system_summary,
)


class TestHealthDiagnostics:
    """Test health reporting subsystem."""

    def test_python_info(self):
        info = get_python_info()
        assert "version" in info
        assert "system" in info
        major, minor = [int(x) for x in info["version"].split(".")[:2]]
        assert (major, minor) >= (3, 10), "Requires Python >= 3.10"

    def test_gpu_info(self):
        gpu = get_gpu_info()
        assert "cuda_available" in gpu
        assert isinstance(gpu["cuda_available"], bool)
        assert "default_device" in gpu
        assert gpu["default_device"] in {"cpu", "cuda:0"}

    def test_package_statuses(self):
        statuses = get_package_statuses()
        assert len(statuses) > 10
        # Verify required packages are evaluated
        installed_names = {s["name"] for s in statuses if s["installed"]}
        assert "fastapi" in installed_names
        assert "streamlit" in installed_names
        assert "torch" in installed_names
        assert "geopandas" in installed_names
        assert "networkx" in installed_names

    def test_directory_health(self):
        dirs = get_directory_health()
        assert "app" in dirs
        assert "cv" in dirs
        assert "ml" in dirs
        assert "gis" in dirs
        assert "routing"
        # All required Phase 1 dirs must exist
        assert dirs["app"] is True
        assert dirs["cv"] is True
        assert dirs["ml"] is True
        assert dirs["gis"] is True
        assert dirs["routing"] is True
        assert dirs["data"] is True
        assert dirs["models"] is True
        assert dirs["scripts"] is True
        assert dirs["tests"] is True
        assert dirs["config"] is True
        assert dirs["notebooks"] is True

    def test_system_summary(self):
        summary = get_system_summary()
        assert summary["project_name"] == "Chennai Flood Access Risk Mapper with DrainWatch AI"
        assert "python" in summary
        assert "gpu" in summary
        assert "packages" in summary
        assert "directories" in summary
