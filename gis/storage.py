"""
Geospatial storage manager.
Ensures processed geospatial layers are stored separately from raw data
in standard formats (GeoParquet, GeoJSON, GeoPackage).
"""
from pathlib import Path
from typing import Optional, Literal
import geopandas as gpd

from config.constants import CRS_WGS84


PROCESSED_GIS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "gis"
RAW_GIS_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def save_processed_layer(
    gdf: gpd.GeoDataFrame,
    layer_name: str,
    file_format: Literal["parquet", "geojson", "gpkg"] = "parquet",
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Save cleaned and preprocessed geospatial data into processed GIS storage.

    Parameters:
        gdf: Cleaned GeoDataFrame
        layer_name: Name of layer (e.g. 'roads_cleaned', 'water_features', 'gcc_zones')
        file_format: Output format ('parquet', 'geojson', 'gpkg')
        output_dir: Destination folder (defaults to data/processed/gis/)

    Returns:
        Path to saved file
    """
    if gdf is None or gdf.empty:
        raise ValueError(f"Cannot save empty GeoDataFrame for layer '{layer_name}'")

    target_dir = output_dir or PROCESSED_GIS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    if file_format == "parquet":
        out_path = target_dir / f"{layer_name}.parquet"
        gdf.to_parquet(out_path)
    elif file_format == "geojson":
        out_path = target_dir / f"{layer_name}.geojson"
        gdf.to_file(out_path, driver="GeoJSON")
    elif file_format == "gpkg":
        out_path = target_dir / f"{layer_name}.gpkg"
        gdf.to_file(out_path, driver="GPKG")
    else:
        raise ValueError(f"Unsupported geospatial format: {file_format}")

    return out_path


def load_processed_layer(
    layer_name: str,
    file_format: Literal["parquet", "geojson", "gpkg"] = "parquet",
    input_dir: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """Load a previously processed GIS layer from processed storage."""
    target_dir = input_dir or PROCESSED_GIS_DIR
    ext = f".{file_format}"
    target_file = target_dir / f"{layer_name}{ext}"

    if not target_file.exists():
        raise FileNotFoundError(f"Processed layer '{layer_name}' not found at {target_file}")

    if file_format == "parquet":
        return gpd.read_parquet(target_file)
    else:
        return gpd.read_file(target_file)
