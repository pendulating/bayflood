#!/usr/bin/env python
"""
Generate topology (elevation) statistics aggregated by census geometry.

This script processes DEM raster data and computes zonal statistics
(min, max, mean elevation) for any census geometry level.

Usage:
    python pp_topology.py --geometry-type ct
    python pp_topology.py --geometry-type cbg
    python pp_topology.py --geometry-type cb
"""

import os 
import argparse
import sys
from pathlib import Path

import rasterio
from rasterio.enums import Resampling
from rasterstats import zonal_stats
import pandas as pd
import geopandas as gpd 

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from geometry_config import GeometryType, get_geometry_paths, get_geometry_config
from logger import setup_logger

logger = setup_logger('pp-topology-subroutine')
logger.setLevel("INFO")


def downsample_raster(input_path: str, output_path: str, downsample_factor: int = 10):
    """
    Downsample a raster for faster processing.
    
    Parameters
    ----------
    input_path : str
        Path to input raster (DEM)
    output_path : str
        Path for downsampled output
    downsample_factor : int
        Factor to downsample by (default: 10)
    """
    logger.info(f"Downsampling raster by factor {downsample_factor}")
    
    with rasterio.open(input_path) as src:
        new_transform = src.transform * src.transform.scale(
            downsample_factor,
            downsample_factor
        )
        new_width = src.width // downsample_factor
        new_height = src.height // downsample_factor
        
        topology = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.bilinear
        )

        new_meta = src.meta.copy()
        new_meta.update({
            "driver": "GTiff",
            "height": new_height,
            "width": new_width,
            "transform": new_transform
        })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, "w", **new_meta) as dst:
            dst.write(topology)
    
    logger.success(f"Downsampled raster saved to {output_path}")
    return output_path


def sample_topology(
    topology_path: str, 
    geometry_path: str, 
    id_column: str,
    output_path: str
) -> pd.DataFrame:
    """
    Compute zonal statistics for topology raster.
    
    Parameters
    ----------
    topology_path : str
        Path to topology raster (GeoTIFF)
    geometry_path : str
        Path to census geometry (GeoJSON)
    id_column : str
        Column name for geography ID
    output_path : str
        Path for output CSV
        
    Returns
    -------
    pd.DataFrame
        DataFrame with topology statistics per geography
    """
    assert topology_path.endswith('.tif'), "topology_path must end in .tif"
    
    logger.info(f"Loading geometry from {geometry_path}")
    gdf = gpd.read_file(geometry_path).to_crs("EPSG:2263")
    logger.info(f"Loaded {len(gdf)} geometries")

    logger.info(f"Computing zonal statistics from {topology_path}")
    summary_stats = zonal_stats(gdf, topology_path)
    
    # Convert to DataFrame
    summary_stats_df = pd.DataFrame(summary_stats)
    
    # Join with geometry ID
    result = gdf[[id_column]].copy()
    result = result.join(summary_stats_df)
    
    # Drop count column, keep min/max/mean
    if 'count' in result.columns:
        result = result.drop(columns='count')
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    
    logger.success(f"Topology statistics saved to {output_path}")
    logger.info(f"  min elevation: {result['min'].min():.1f} - {result['min'].max():.1f}")
    logger.info(f"  mean elevation: {result['mean'].min():.1f} - {result['mean'].max():.1f}")
    logger.info(f"  max elevation: {result['max'].min():.1f} - {result['max'].max():.1f}")
    
    return result


def generate_topology(geometry_type: GeometryType, base_dir: Path) -> Path:
    """
    Generate topology statistics for a geometry type.
    
    Parameters
    ----------
    geometry_type : GeometryType
        Census geometry level
    base_dir : Path
        Base directory of the project
        
    Returns
    -------
    Path
        Path to generated topology CSV
    """
    paths = get_geometry_paths(geometry_type, str(base_dir))
    config = get_geometry_config(geometry_type)
    
    logger.info(f"Generating topology for {config.display_name}s")
    
    # Input paths
    dem_path = base_dir / 'aggregation' / 'geo' / 'data' / 'DEM_LiDAR_1ft_2010_Improved_NYC_int.tif'
    downsampled_path = base_dir / 'aggregation' / 'geo' / 'data' / 'processed' / 'topology_nyc_downsampled.tif'
    
    # Check for DEM
    if not dem_path.exists():
        raise FileNotFoundError(
            f"DEM raster not found at {dem_path}\n"
            "Download from: https://data.cityofnewyork.us/City-Government/1-foot-Digital-Elevation-Model-DEM-Integer-Raster/7kuu-zah7"
        )
    
    # Downsample if needed (reuse if exists)
    if not downsampled_path.exists():
        logger.info("Downsampled raster not found, creating...")
        downsample_raster(str(dem_path), str(downsampled_path), downsample_factor=10)
    else:
        logger.info(f"Using existing downsampled raster: {downsampled_path}")
    
    # Get geometry path
    geojson_path = paths.aggregation_geojson_path
    if not geojson_path.exists():
        geojson_path = paths.geojson_path
    
    # Output path
    output_path = paths.topology_path
    
    # Generate topology
    sample_topology(
        topology_path=str(downsampled_path),
        geometry_path=str(geojson_path),
        id_column=config.id_column,
        output_path=str(output_path)
    )
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate topology statistics by census geometry"
    )
    parser.add_argument(
        '--geometry-type',
        type=str,
        required=True,
        choices=['ct', 'cbg', 'cb'],
        help='Census geometry type: ct (tract), cbg (block group), cb (block)'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=None,
        help='Base directory (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir) if args.base_dir else Path(__file__).parent.parent.parent
    geometry_type = GeometryType(args.geometry_type)
    
    try:
        output_path = generate_topology(geometry_type, base_dir)
        print(f"\nTopology generated: {output_path}")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
