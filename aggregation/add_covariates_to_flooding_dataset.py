#!/usr/bin/env python
"""
Add external covariates to a flooding dataset.

This script adds the same covariates that context_df.ipynb adds to CT:
- Elevation (topology)
- FloodNet sensors
- Catch basins
- 311 clogged catch basin complaints
- DEP stormwater flooding areas

Usage:
    python add_covariates_to_flooding_dataset.py --geometry-type cbg
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from geometry_config import GeometryType, get_geometry_paths, get_geometry_config
from logger import setup_logger

logger = setup_logger("add-covariates")
logger.setLevel("INFO")

WGS = 'EPSG:4326'
PROJ = 'EPSG:2263'


def add_elevation_covariates(gdf: pd.DataFrame, base_dir: Path, geometry_type: GeometryType, id_column: str) -> pd.DataFrame:
    """Add elevation/topology covariates."""
    paths = get_geometry_paths(geometry_type, str(base_dir))
    topology_path = paths.topology_path
    
    if topology_path.exists():
        logger.info(f"Loading topology from {topology_path}")
        topology = pd.read_csv(topology_path)
        
        if id_column in topology.columns:
            topology[id_column] = topology[id_column].astype(str)
            # Rename columns to have ft_elevation_ prefix
            rename_map = {c: f'ft_elevation_{c}' for c in ['min', 'max', 'mean'] if c in topology.columns}
            topology = topology.rename(columns=rename_map)
            
            # Only keep id and elevation columns
            cols_to_keep = [id_column] + [c for c in topology.columns if c.startswith('ft_elevation_')]
            topology = topology[cols_to_keep]
            
            gdf = gdf.merge(topology, on=id_column, how='left')
            logger.info(f"Added elevation columns: {[c for c in topology.columns if c.startswith('ft_elevation_')]}")
    else:
        # Set default values (city-wide means from CT data)
        logger.warning(f"Topology file not found at {topology_path}, using default values")
        gdf['ft_elevation_min'] = 10.0  # Approximate NYC mean
        gdf['ft_elevation_mean'] = 30.0
        gdf['ft_elevation_max'] = 50.0
    
    return gdf


def add_floodnet_covariates(gdf: gpd.GeoDataFrame, gdf_geo: gpd.GeoDataFrame, 
                            base_dir: Path, id_column: str) -> gpd.GeoDataFrame:
    """Add FloodNet sensor covariates."""
    floodnet_path = base_dir / 'aggregation' / 'flooding' / 'static' / 'current_floodnet_sensors.csv'
    
    if floodnet_path.exists():
        logger.info(f"Loading FloodNet sensors from {floodnet_path}")
        sensors = pd.read_csv(floodnet_path)
        sensors_geo = sensors.groupby('deployment_id').first()
        sensors_gdf = gpd.GeoDataFrame(
            sensors_geo,
            geometry=gpd.points_from_xy(sensors_geo['longitude'], sensors_geo['latitude']),
            crs=WGS
        ).to_crs(PROJ)
        
        # Count sensors per area
        joined = gpd.sjoin(gdf_geo, sensors_gdf, how='left', predicate='contains')
        sensor_counts = joined.groupby(id_column).size()
        gdf['n_floodnet_sensors'] = gdf[id_column].map(sensor_counts).fillna(0).astype(int)
        logger.info(f"Added n_floodnet_sensors: {gdf['n_floodnet_sensors'].sum()} total")
    else:
        logger.warning(f"FloodNet file not found, setting n_floodnet_sensors=0")
        gdf['n_floodnet_sensors'] = 0
    
    return gdf


def add_catch_basin_covariates(gdf: gpd.GeoDataFrame, gdf_geo: gpd.GeoDataFrame,
                                base_dir: Path, id_column: str) -> gpd.GeoDataFrame:
    """Add catch basin covariates."""
    cb_path = base_dir / 'aggregation' / 'flooding' / 'static' / 'catch_basins_nyc.geojson'
    
    if cb_path.exists():
        logger.info(f"Loading catch basins from {cb_path}")
        catch_basins = gpd.read_file(cb_path).to_crs(PROJ)
        
        # Count catch basins per area
        joined = gpd.sjoin(gdf_geo, catch_basins, how='left', predicate='contains')
        cb_counts = joined.groupby(id_column).size()
        gdf['n_catch_basins'] = gdf[id_column].map(cb_counts).fillna(0).astype(int)
        gdf['catch_basin_density'] = gdf['n_catch_basins'] / gdf['area']
        logger.info(f"Added catch basin covariates: {gdf['n_catch_basins'].sum()} total")
    else:
        logger.warning(f"Catch basins file not found, setting n_catch_basins=0")
        gdf['n_catch_basins'] = 0
        gdf['catch_basin_density'] = 0
    
    return gdf


def add_clogged_cb_covariates(gdf: gpd.GeoDataFrame, gdf_geo: gpd.GeoDataFrame,
                               base_dir: Path, id_column: str) -> gpd.GeoDataFrame:
    """Add clogged catch basin 311 complaint covariates."""
    cb311_path = base_dir / 'aggregation' / 'flooding' / 'data' / 'nyc311_clogged_cb_jun_sep.csv'
    
    if cb311_path.exists():
        logger.info(f"Loading clogged CB complaints from {cb311_path}")
        clogged = pd.read_csv(cb311_path).dropna(subset=['latitude', 'longitude'])
        
        # Calculate days clogged
        clogged['created_date'] = pd.to_datetime(clogged['created_date'])
        clogged['closed_date'] = pd.to_datetime(clogged['closed_date'])
        end_date = pd.Timestamp('2023-09-29')
        clogged['effective_closed'] = clogged['closed_date'].fillna(end_date).clip(upper=end_date)
        clogged['days_clogged'] = (clogged['effective_closed'] - clogged['created_date']).dt.days.clip(lower=0)
        
        clogged_gdf = gpd.GeoDataFrame(
            clogged,
            geometry=gpd.points_from_xy(clogged['longitude'], clogged['latitude']),
            crs=WGS
        ).to_crs(PROJ)
        
        # Join and aggregate
        joined = gpd.sjoin(clogged_gdf, gdf_geo[[id_column, 'geometry']], how='left', predicate='within')
        
        cb_days = joined.groupby(id_column)['days_clogged'].sum()
        avg_resolution = joined.groupby(id_column)['days_clogged'].mean()
        city_median = clogged['days_clogged'].median()
        
        gdf['cb_days_clogged'] = gdf[id_column].map(cb_days).fillna(0)
        gdf['has_clogged_cb_complaint'] = gdf[id_column].map(avg_resolution).notna().astype(int)
        gdf['cb_avg_resolution_time'] = gdf[id_column].map(avg_resolution).fillna(city_median)
        
        logger.info(f"Added clogged CB covariates: {gdf['cb_days_clogged'].sum():.0f} total days")
    else:
        logger.warning(f"Clogged CB file not found, setting defaults")
        gdf['cb_days_clogged'] = 0
        gdf['has_clogged_cb_complaint'] = 0
        gdf['cb_avg_resolution_time'] = 0
    
    return gdf


def add_dep_stormwater_covariates(gdf: gpd.GeoDataFrame, gdf_geo: gpd.GeoDataFrame,
                                   base_dir: Path, id_column: str) -> gpd.GeoDataFrame:
    """Add DEP stormwater flooding covariates."""
    dep_path = base_dir / 'aggregation' / 'flooding' / 'static' / 'dep_stormwater_moderate_current' / 'data.gdb'
    
    if dep_path.exists():
        logger.info(f"Loading DEP stormwater from {dep_path}")
        dep = gpd.read_file(str(dep_path)).to_crs(PROJ)
        
        # Flatten multipolygons
        polygons = {}
        for i, row in dep.iterrows():
            for idx, polygon in enumerate(row['geometry'].geoms):
                polygons[f"{row['Flooding_Category']}_{idx}"] = polygon
        
        dep_flat = gpd.GeoDataFrame(polygons, index=['geometry']).T
        dep_flat.set_geometry('geometry', inplace=True)
        dep_flat.crs = dep.crs
        dep_flat['Flooding_Category'] = dep_flat.index.str.split('_').str[0].astype(int)
        
        for category in [1, 2]:
            dep_cat = dep_flat[dep_flat['Flooding_Category'] <= category]
            area_col = f'dep_moderate_{category}_area'
            frac_col = f'dep_moderate_{category}_frac'
            
            overlay = gpd.overlay(gdf_geo, dep_cat, how='intersection')
            area_by_geo = overlay.groupby(id_column)['geometry'].apply(
                lambda geom: geom.area.sum()
            )
            gdf[area_col] = gdf[id_column].map(area_by_geo).fillna(0)
            gdf[frac_col] = gdf[area_col] / gdf['area']
        
        logger.info(f"Added DEP stormwater covariates")
    else:
        logger.warning(f"DEP stormwater file not found, setting defaults")
        gdf['dep_moderate_1_area'] = 0
        gdf['dep_moderate_1_frac'] = 0
        gdf['dep_moderate_2_area'] = 0
        gdf['dep_moderate_2_frac'] = 0
    
    return gdf


def add_311_flooding_covariates(gdf: gpd.GeoDataFrame, gdf_geo: gpd.GeoDataFrame,
                                 base_dir: Path, id_column: str) -> gpd.GeoDataFrame:
    """Add 311 flooding complaint covariates."""
    nyc311_path = base_dir / 'aggregation' / 'flooding' / 'data' / 'nyc311_flooding_sep29.csv'
    
    if nyc311_path.exists():
        logger.info(f"Loading 311 complaints from {nyc311_path}")
        nyc311 = pd.read_csv(nyc311_path).dropna(subset=['latitude', 'longitude'])
        nyc311_gdf = gpd.GeoDataFrame(
            nyc311,
            geometry=gpd.points_from_xy(nyc311['longitude'], nyc311['latitude']),
            crs=WGS
        ).to_crs(PROJ)
        
        # Count by area
        joined = gpd.sjoin(nyc311_gdf, gdf_geo[[id_column, 'geometry']], how='left', predicate='within')
        counts = joined.groupby(id_column).size()
        gdf['n_311_reports'] = gdf[id_column].map(counts).fillna(0).astype(int)
        logger.info(f"Added n_311_reports: {gdf['n_311_reports'].sum()} total")
    else:
        logger.warning(f"311 file not found, setting n_311_reports=0")
        gdf['n_311_reports'] = 0
    
    return gdf


def add_covariates(geometry_type: GeometryType, base_dir: Path) -> Path:
    """Add all covariates to the flooding dataset."""
    paths = get_geometry_paths(geometry_type, str(base_dir))
    config = paths.config
    id_column = config.id_column
    
    logger.info(f"Adding covariates to {config.display_name} flooding dataset")
    
    # Load flooding dataset
    flooding_path = paths.flooding_dataset_path
    if not flooding_path.exists():
        raise FileNotFoundError(f"Flooding dataset not found: {flooding_path}")
    
    df = pd.read_csv(flooding_path)
    df[id_column] = df[id_column].astype(str)
    logger.info(f"Loaded {len(df)} rows from {flooding_path}")
    
    # Load geometry for spatial operations
    geojson_path = paths.geojson_path
    gdf_geo = gpd.read_file(str(geojson_path)).to_crs(PROJ)
    gdf_geo[id_column] = gdf_geo[id_column].astype(str)
    gdf_geo['area'] = gdf_geo.geometry.area
    
    # Convert df to GeoDataFrame for merging
    gdf = df.copy()
    gdf['area'] = gdf[id_column].map(gdf_geo.set_index(id_column)['area'])
    
    # Add all covariates
    gdf = add_elevation_covariates(gdf, base_dir, geometry_type, id_column)
    gdf = add_floodnet_covariates(gdf, gdf_geo, base_dir, id_column)
    gdf = add_catch_basin_covariates(gdf, gdf_geo, base_dir, id_column)
    gdf = add_clogged_cb_covariates(gdf, gdf_geo, base_dir, id_column)
    gdf = add_dep_stormwater_covariates(gdf, gdf_geo, base_dir, id_column)
    gdf = add_311_flooding_covariates(gdf, gdf_geo, base_dir, id_column)
    
    # Save
    output_path = flooding_path  # Overwrite original
    gdf.to_csv(output_path, index=False)
    logger.success(f"Saved updated dataset to {output_path}")
    
    # Print summary
    covariate_cols = ['ft_elevation_min', 'ft_elevation_mean', 'n_floodnet_sensors',
                      'n_catch_basins', 'catch_basin_density', 'cb_days_clogged',
                      'cb_avg_resolution_time', 'has_clogged_cb_complaint',
                      'dep_moderate_1_frac', 'dep_moderate_2_frac', 'n_311_reports']
    
    print("\n=== Covariate Summary ===")
    for col in covariate_cols:
        if col in gdf.columns:
            print(f"{col}: mean={gdf[col].mean():.4f}, sum={gdf[col].sum():.2f}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Add covariates to flooding dataset")
    parser.add_argument(
        '--geometry-type',
        type=str,
        required=True,
        choices=['ct', 'cbg', 'cb'],
        help='Census geometry type'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=None,
        help='Base directory (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir) if args.base_dir else Path(__file__).parent.parent
    geometry_type = GeometryType(args.geometry_type)
    
    add_covariates(geometry_type, base_dir)


if __name__ == '__main__':
    main()

