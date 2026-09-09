"""
Unit tests for the Chennai rainfall data pipeline.
Verifies:
- All 5 accumulation windows: 15min, 1h, 3h, 6h, 24h
- Real rainfall ingestion & station registry
- Cleaning and meteorological quality control
- Temporal rolling aggregation
- Missing-data imputation (temporal & spatial fallback)
- Spatial association with Chennai road segments (IDW & nearest)
- Scenario input interface (50mm, 75mm, 100mm, 150mm, 200mm, 250mm, custom)
- Strict distinction between REAL and SIMULATED rainfall
"""
import pytest
from datetime import datetime, timedelta
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString

from config.constants import CRS_WGS84
from rainfall.models import (
    RainfallSourceType,
    RainfallStation,
    RainfallReading,
    RainfallAggregates,
)
from rainfall.ingestion import RealRainfallIngestor, CHENNAI_MET_STATIONS
from rainfall.cleaning import clean_rainfall_readings, readings_to_dataframe
from rainfall.aggregation import (
    aggregate_station_readings,
    compute_rolling_aggregates_timeseries,
)
from rainfall.missing import impute_temporal_gaps, spatial_fallback_impute
from rainfall.spatial import associate_rainfall_with_roads
from rainfall.scenarios import (
    RainfallScenarioBuilder,
    STANDARD_SCENARIOS,
    DISTRIBUTION_PROFILES,
)


# ─── 1. REAL vs SIMULATED Distinction ────────────────────────────────────────

class TestRainfallSourceDistinction:
    """Verify strict separation between REAL and SIMULATED rainfall."""

    def test_source_type_enum_values(self):
        assert RainfallSourceType.REAL.value == "REAL"
        assert RainfallSourceType.SIMULATED.value == "SIMULATED"

    def test_real_readings_are_tagged_real(self):
        reading = RainfallReading(
            station_id="IMD_43279",
            timestamp=datetime(2026, 9, 9, 12, 0),
            precipitation_mm=15.5,
            source_type=RainfallSourceType.REAL,
        )
        assert reading.source_type == RainfallSourceType.REAL

    def test_simulated_scenarios_are_tagged_simulated(self):
        scenario_aggs = RainfallScenarioBuilder.build_aggregates_from_total(100.0)
        assert scenario_aggs.source_type == RainfallSourceType.SIMULATED


# ─── 2. Ingestion Tests ──────────────────────────────────────────────────────

class TestRainfallIngestion:
    """Verify real station data ingestion without fabrication."""

    def test_chennai_weather_stations_registry(self):
        ingestor = RealRainfallIngestor()
        stations = ingestor.get_registered_stations()
        assert len(stations) >= 5

        station_names = [s.name for s in stations]
        assert any("Nungambakkam" in name for name in station_names)
        assert any("Meenambakkam" in name for name in station_names)

        # Coordinate check inside Chennai bounding box
        for s in stations:
            assert 12.80 <= s.latitude <= 13.30
            assert 80.00 <= s.longitude <= 80.35

    def test_load_from_csv(self, tmp_path):
        csv_file = tmp_path / "nungambakkam_rain.csv"
        csv_content = (
            "timestamp,station_id,precipitation_mm\n"
            "2026-09-09 10:00:00,IMD_43279,5.0\n"
            "2026-09-09 10:15:00,IMD_43279,12.5\n"
            "2026-09-09 10:30:00,IMD_43279,8.0\n"
        )
        csv_file.write_text(csv_content)

        ingestor = RealRainfallIngestor()
        readings = ingestor.load_from_csv(csv_file)

        assert len(readings) == 3
        assert all(r.source_type == RainfallSourceType.REAL for r in readings)
        assert readings[1].precipitation_mm == 12.5

    def test_missing_csv_raises_file_not_found(self):
        ingestor = RealRainfallIngestor()
        with pytest.raises(FileNotFoundError):
            ingestor.load_from_csv("non_existent_rain.csv")

    def test_missing_api_key_refuses_to_fabricate_live_data(self):
        ingestor = RealRainfallIngestor()
        with pytest.raises(ValueError, match="Do not fabricate real rainfall observations"):
            ingestor.fetch_live_api(api_key=None)


# ─── 3. Cleaning & Quality Control Tests ─────────────────────────────────────

class TestRainfallCleaning:
    """Verify physical quality checks and outlier handling."""

    def test_negative_values_corrected(self):
        bad_reading = RainfallReading(
            station_id="IMD_43279",
            timestamp=datetime(2026, 9, 9, 12, 0),
            precipitation_mm=0.0,
            source_type=RainfallSourceType.REAL,
        )
        # Directly test negative correction in cleaner
        raw = [bad_reading]
        raw[0].__dict__["precipitation_mm"] = -10.0  # bypass pydantic validator to test cleaning defense

        cleaned = clean_rainfall_readings(raw)
        assert cleaned[0].precipitation_mm == 0.0
        assert cleaned[0].quality_flag == "NEGATIVE_CORRECTED"

    def test_extreme_spike_flagged(self):
        spike = RainfallReading(
            station_id="IMD_43279",
            timestamp=datetime(2026, 9, 9, 12, 0),
            precipitation_mm=150.0,  # > 120mm in 15 min is suspect
            source_type=RainfallSourceType.REAL,
        )
        cleaned = clean_rainfall_readings([spike])
        assert cleaned[0].quality_flag == "SUSPECT_EXTREME"

    def test_deduplication_and_sorting(self):
        t1 = datetime(2026, 9, 9, 10, 15)
        t2 = datetime(2026, 9, 9, 10, 0)
        readings = [
            RainfallReading(station_id="S1", timestamp=t1, precipitation_mm=5.0, source_type=RainfallSourceType.REAL),
            RainfallReading(station_id="S1", timestamp=t2, precipitation_mm=2.0, source_type=RainfallSourceType.REAL),
            RainfallReading(station_id="S1", timestamp=t1, precipitation_mm=5.0, source_type=RainfallSourceType.REAL),  # duplicate
        ]
        cleaned = clean_rainfall_readings(readings)
        assert len(cleaned) == 2
        assert cleaned[0].timestamp == t2
        assert cleaned[1].timestamp == t1


# ─── 4. Temporal Aggregation Tests ───────────────────────────────────────────

class TestTemporalAggregation:
    """Verify all 5 required accumulation windows: 15min, 1h, 3h, 6h, 24h."""

    def test_all_five_windows_aggregated_accurately(self):
        # Generate 24 hours of 15-minute readings (96 intervals)
        base_time = datetime(2026, 9, 9, 0, 0)
        readings: List[RainfallReading] = []

        for i in range(96):
            t = base_time + timedelta(minutes=15 * i)
            # Constant 1 mm every 15 minutes = 4 mm/hr
            readings.append(
                RainfallReading(
                    station_id="TEST_ST",
                    timestamp=t,
                    precipitation_mm=1.0,
                    source_type=RainfallSourceType.REAL,
                )
            )

        # End of day reference
        as_of = base_time + timedelta(hours=24) - timedelta(minutes=1)
        aggs = aggregate_station_readings(readings, as_of=as_of)

        # 1 interval
        assert aggs.rainfall_15min == 1.0
        # 4 intervals (1 hr)
        assert aggs.rainfall_1h == 4.0
        # 12 intervals (3 hrs)
        assert aggs.rainfall_3h == 12.0
        # 24 intervals (6 hrs)
        assert aggs.rainfall_6h == 24.0
        # 96 intervals (24 hrs)
        assert aggs.rainfall_24h == 96.0
        assert aggs.source_type == RainfallSourceType.REAL

    def test_rolling_aggregates_timeseries_structure(self):
        readings = [
            RainfallReading(station_id="S1", timestamp=datetime(2026, 9, 9, 10, 0), precipitation_mm=2.0, source_type=RainfallSourceType.REAL),
            RainfallReading(station_id="S1", timestamp=datetime(2026, 9, 9, 10, 15), precipitation_mm=3.0, source_type=RainfallSourceType.REAL),
        ]
        df = compute_rolling_aggregates_timeseries(readings)
        for col in ["rainfall_15min", "rainfall_1h", "rainfall_3h", "rainfall_6h", "rainfall_24h"]:
            assert col in df.columns


# ─── 5. Missing Data Handling Tests ──────────────────────────────────────────

class TestMissingDataHandling:
    """Verify gap filling and spatial fallback."""

    def test_impute_temporal_gap(self):
        t0 = datetime(2026, 9, 9, 10, 0)
        # Gap between 10:00 and 10:45 (missing 10:15 and 10:30)
        readings = [
            RainfallReading(station_id="S1", timestamp=t0, precipitation_mm=4.0, source_type=RainfallSourceType.REAL),
            RainfallReading(station_id="S1", timestamp=t0 + timedelta(minutes=45), precipitation_mm=8.0, source_type=RainfallSourceType.REAL),
        ]
        imputed = impute_temporal_gaps(readings, expected_freq="15min", max_gap_hours=2.0)

        # Should now have 4 readings: 10:00, 10:15, 10:30, 10:45
        assert len(imputed) == 4
        imputed_flags = [r.quality_flag for r in imputed]
        assert "IMPUTED_TEMPORAL" in imputed_flags

    def test_spatial_fallback_when_station_offline(self):
        st_failed = RainfallStation(station_id="OFFLINE_ST", name="Offline Station", latitude=13.05, longitude=80.25)
        st_donor = RainfallStation(station_id="DONOR_ST", name="Active Donor", latitude=13.06, longitude=80.26)

        donor_readings = [
            RainfallReading(
                station_id="DONOR_ST",
                timestamp=datetime(2026, 9, 9, 12, 0),
                precipitation_mm=18.0,
                source_type=RainfallSourceType.REAL,
            )
        ]

        fallback = spatial_fallback_impute(
            target_station=st_failed,
            target_readings=[],  # failed station has zero readings
            donor_stations={"DONOR_ST": st_donor},
            donor_readings={"DONOR_ST": donor_readings},
        )

        assert len(fallback) == 1
        assert fallback[0].station_id == "OFFLINE_ST"
        assert fallback[0].precipitation_mm == 18.0
        assert "SPATIAL_SUBSTITUTION" in fallback[0].quality_flag


# ─── 6. Spatial Association with Road Segments Tests ─────────────────────────

class TestSpatialAssociation:
    """Verify association of rainfall metrics across Chennai road network."""

    def test_associate_rainfall_with_roads_idw(self):
        # Create road segments in Chennai
        roads = gpd.GeoDataFrame(
            [
                {"segment_id": "road_nungam", "geometry": LineString([(80.23, 13.06), (80.24, 13.07)])},
                {"segment_id": "road_meenam", "geometry": LineString([(80.16, 12.98), (80.17, 12.99)])},
            ],
            crs=CRS_WGS84,
        )

        st_nungam = RainfallStation(station_id="ST_NUN", name="Nungambakkam", latitude=13.0674, longitude=80.2376)
        st_meenam = RainfallStation(station_id="ST_MEE", name="Meenambakkam", latitude=12.9900, longitude=80.1693)

        # High rainfall at Nungam, lower at Meenam
        aggs_nungam = RainfallAggregates(
            rainfall_15min=20.0, rainfall_1h=50.0, rainfall_3h=80.0, rainfall_6h=110.0, rainfall_24h=160.0,
            source_type=RainfallSourceType.REAL,
        )
        aggs_meenam = RainfallAggregates(
            rainfall_15min=5.0, rainfall_1h=15.0, rainfall_3h=25.0, rainfall_6h=40.0, rainfall_24h=60.0,
            source_type=RainfallSourceType.REAL,
        )

        station_aggs = {"ST_NUN": aggs_nungam, "ST_MEE": aggs_meenam}
        stations_info = {"ST_NUN": st_nungam, "ST_MEE": st_meenam}

        enriched = associate_rainfall_with_roads(roads, station_aggs, stations_info, method="idw")

        # Verify columns exist
        for col in ["rainfall_15min", "rainfall_1h", "rainfall_3h", "rainfall_6h", "rainfall_24h", "rainfall_source_type"]:
            assert col in enriched.columns

        assert all(enriched["rainfall_source_type"] == "REAL")

        # Road near Nungambakkam should have higher rainfall than road near Meenambakkam
        r_nun = enriched[enriched["segment_id"] == "road_nungam"].iloc[0]
        r_mee = enriched[enriched["segment_id"] == "road_meenam"].iloc[0]
        assert r_nun["rainfall_24h"] > r_mee["rainfall_24h"]


# ─── 7. Scenario Input Interface Tests ───────────────────────────────────────

class TestScenarioInterface:
    """Verify arbitrary rainfall scenarios (50, 75, 100, 150, 200, 250mm, custom)."""

    @pytest.mark.parametrize("depth_mm", [50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 185.7])
    def test_arbitrary_scenario_generation(self, depth_mm):
        aggs = RainfallScenarioBuilder.build_aggregates_from_total(depth_mm, distribution="MONSOON_BURST")

        assert aggs.rainfall_24h == depth_mm
        assert aggs.rainfall_6h <= aggs.rainfall_24h
        assert aggs.rainfall_3h <= aggs.rainfall_6h
        assert aggs.rainfall_1h <= aggs.rainfall_3h
        assert aggs.rainfall_15min <= aggs.rainfall_1h
        assert aggs.source_type == RainfallSourceType.SIMULATED

    def test_apply_scenario_to_roads_uniform(self):
        roads = gpd.GeoDataFrame(
            [
                {"segment_id": "r1", "geometry": LineString([(80.24, 13.04), (80.25, 13.05)])},
                {"segment_id": "r2", "geometry": LineString([(80.22, 13.01), (80.23, 13.02)])},
            ],
            crs=CRS_WGS84,
        )

        result = RainfallScenarioBuilder.apply_scenario_to_roads(roads, total_24h_mm=200.0)

        assert all(result["rainfall_24h"] == 200.0)
        assert all(result["rainfall_source_type"] == "SIMULATED")
        assert "SIMULATED_200MM" in result.iloc[0]["scenario_name"]

    def test_spatial_hotspot_scenario(self):
        roads = gpd.GeoDataFrame(
            [
                {"segment_id": "south_road", "geometry": LineString([(80.22, 12.95), (80.23, 12.96)])},  # South Chennai
                {"segment_id": "north_road", "geometry": LineString([(80.22, 13.15), (80.23, 13.16)])},  # North Chennai
            ],
            crs=CRS_WGS84,
        )

        result = RainfallScenarioBuilder.apply_scenario_to_roads(
            roads, total_24h_mm=100.0, hotspot="SOUTH_CHENNAI"
        )

        s_val = result[result["segment_id"] == "south_road"].iloc[0]["rainfall_24h"]
        n_val = result[result["segment_id"] == "north_road"].iloc[0]["rainfall_24h"]

        # South should receive amplified rainfall relative to North in a South hotspot scenario
        assert s_val > n_val
        assert s_val == 125.0
        assert n_val == 85.0
        assert all(result["rainfall_source_type"] == "SIMULATED")
