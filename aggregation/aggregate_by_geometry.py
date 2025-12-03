#!/usr/bin/env python
"""
Parameterized aggregation script for generating context dataframes.

This script aggregates flooding data and covariates to different census
geography levels (Census Tracts, Block Groups, or Blocks).

Usage:
    python aggregate_by_geometry.py --geometry-type ct
    python aggregate_by_geometry.py --geometry-type cbg
    python aggregate_by_geometry.py --geometry-type cb
"""

import argparse
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from geometry_config import (
    GeometryType,
    GeometryPaths,
    get_geometry_paths,
    get_geometry_config,
)
from logger import setup_logger

# Constants
WGS = 'EPSG:4326'
PROJ = 'EPSG:2263'


def load_geometry_data(paths: GeometryPaths, logger) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load the base geometry data.
    
    Returns tuple of (attribute_df, geometry_gdf)
    """
    config = paths.config
    geojson_path = paths.aggregation_geojson_path
    
    if not geojson_path.exists():
        # Try the data directory path
        geojson_path = paths.geojson_path
    
    logger.info(f"Loading {config.display_name} shapefile from {geojson_path}")
    
    gdf = gpd.read_file(geojson_path)
    
    # Standard columns to drop if present
    TO_DROP = ['OBJECTID', 'BoroCode', 'CT2020', 'CDEligibil', 'NTA2020', 
               'CDTA2020', 'Shape__Area', 'Shape__Length', 'geometry']
    cols_to_drop = [c for c in TO_DROP if c in gdf.columns]
    
    # Create attribute df (no geometry)
    attr_df = gdf.drop(columns=cols_to_drop, errors='ignore')
    attr_df = attr_df.set_index(config.id_column).astype(str)
    
    logger.info(f"Loaded {len(attr_df)} {config.display_name}s")
    
    # Create geometry gdf (with geometry)
    geo_gdf = gpd.read_file(geojson_path).to_crs(PROJ)[[config.id_column, 'geometry']]
    geo_gdf['area'] = geo_gdf.area
    
    return attr_df, geo_gdf


def load_flooding_data(paths: GeometryPaths, attr_df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Load flooding model input data if available.
    """
    flooding_path = paths.flooding_dataset_path
    
    if flooding_path.exists():
        logger.info(f"Loading flooding data from {flooding_path}")
        flooding_data = pd.read_csv(flooding_path)
        
        config = paths.config
        flooding_data[config.id_column] = flooding_data[config.id_column].astype(str)
        
        # Standard columns to drop
        TO_DROP = ['CTLabel', 'CDTANAME', 'BoroName', 'BoroCode', 'CT2020', 
                   'BoroCT2020', 'CDEligibil', 'NTAName', 'PUMA', 'geometry']
        cols_to_drop = [c for c in TO_DROP if c in flooding_data.columns]
        flooding_data = flooding_data.drop(columns=cols_to_drop, errors='ignore')
        
        attr_df = attr_df.join(flooding_data.set_index(config.id_column))
        logger.info(f"Merged flooding data")
    else:
        logger.warning(f"Flooding data not found at {flooding_path}")
    
    return attr_df


def load_floodnet_sensors(base_dir: Path, geo_gdf: gpd.GeoDataFrame, 
                          attr_df: pd.DataFrame, id_column: str, logger) -> pd.DataFrame:
    """Load and aggregate FloodNet sensor data."""
    sensor_path = base_dir / 'aggregation' / 'flooding' / 'static' / 'current_floodnet_sensors.csv'
    
    if not sensor_path.exists():
        logger.warning(f"FloodNet sensors not found at {sensor_path}")
        return attr_df
    
    sensors = pd.read_csv(sensor_path, engine='pyarrow')
    sensors_geo = sensors.groupby('deployment_id').first()
    sensors_geo = gpd.GeoDataFrame(
        sensors_geo, 
        geometry=gpd.points_from_xy(sensors_geo['longitude'], sensors_geo['latitude']),
        crs=WGS
    ).to_crs(PROJ)
    
    # Remove duplicate sensors within 1ft of each other
    to_drop = []
    for i, row in sensors_geo.iterrows():
        if i in to_drop:
            continue
        for j, row2 in sensors_geo.iterrows():
            if i == j:
                continue
            if row.geometry.distance(row2.geometry) < 1:
                to_drop.append(j)
    sensors_geo = sensors_geo.drop(to_drop)
    
    logger.info(f"Loaded {len(sensors_geo)} FloodNet sensors")
    
    # Count sensors per area
    attr_df['n_floodnet_sensors'] = (
        gpd.sjoin(geo_gdf, sensors_geo)
        .groupby(id_column).size()
        .reindex(attr_df.index).fillna(0)
    )
    
    return attr_df


def load_catch_basins(base_dir: Path, geo_gdf: gpd.GeoDataFrame,
                      attr_df: pd.DataFrame, id_column: str, logger) -> pd.DataFrame:
    """Load and aggregate catch basin data."""
    cb_path = base_dir / 'aggregation' / 'flooding' / 'static' / 'catch_basins_nyc.geojson'
    
    if not cb_path.exists():
        logger.warning(f"Catch basins not found at {cb_path}")
        return attr_df
    
    catch_basins = gpd.read_file(cb_path).to_crs(PROJ)
    logger.info(f"Loaded {len(catch_basins)} catch basins")
    
    # Count catch basins per area
    attr_df['n_catch_basins'] = (
        gpd.sjoin(geo_gdf, catch_basins)
        .groupby(id_column).size()
        .reindex(attr_df.index).fillna(0)
    )
    attr_df['catch_basin_density'] = attr_df['n_catch_basins'] / attr_df['area']
    
    return attr_df


def load_311_complaints(base_dir: Path, geo_gdf: gpd.GeoDataFrame,
                        attr_df: pd.DataFrame, id_column: str, logger) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Load and aggregate 311 flood complaints."""
    complaints_path = base_dir / 'aggregation' / 'flooding' / 'data' / 'nyc311_flooding_sep29.csv'
    
    if not complaints_path.exists():
        logger.warning(f"311 complaints not found at {complaints_path}")
        return attr_df, geo_gdf
    
    nyc311 = pd.read_csv(complaints_path).dropna(subset=['latitude', 'longitude'])
    nyc311 = gpd.GeoDataFrame(
        nyc311,
        geometry=gpd.points_from_xy(nyc311.longitude, nyc311.latitude),
        crs=WGS
    ).to_crs(PROJ)
    
    logger.info(f"Loaded {len(nyc311)} 311 complaints")
    
    # Count complaints by descriptor type
    for descriptor in nyc311['descriptor'].unique():
        desc = descriptor.split('(')[0].strip().lower().replace(' ', '_') + '_311c'
        geo_gdf[desc] = geo_gdf['geometry'].apply(
            lambda x: nyc311[nyc311['descriptor'] == descriptor].within(x).sum()
        )
    
    # Total 311 reports
    cols_311 = [c for c in geo_gdf.columns if '_311c' in c]
    geo_gdf['n_311_reports'] = geo_gdf[cols_311].sum(axis=1)
    
    return attr_df, geo_gdf


def load_dep_stormwater(base_dir: Path, geo_gdf: gpd.GeoDataFrame,
                        attr_df: pd.DataFrame, id_column: str, logger) -> pd.DataFrame:
    """Load and aggregate DEP stormwater flooding areas."""
    dep_path = base_dir / 'aggregation' / 'flooding' / 'static' / 'dep_stormwater_moderate_current' / 'data.gdb'
    
    if not dep_path.exists():
        logger.warning(f"DEP stormwater data not found at {dep_path}")
        return attr_df
    
    dep_moderate = gpd.read_file(dep_path).to_crs(PROJ)
    logger.info(f"Loaded DEP stormwater data")
    
    # Flatten multipolygons
    polygons = {}
    for i, row in dep_moderate.iterrows():
        for idx, polygon in enumerate(row['geometry'].geoms):
            polygons[f'{row["Flooding_Category"]}_{idx}'] = polygon
    
    dep_flat = gpd.GeoDataFrame(polygons, index=['geometry']).T
    dep_flat.set_geometry('geometry', inplace=True)
    dep_flat.crs = dep_moderate.crs
    dep_flat['Flooding_Category'] = dep_flat.index.str.split('_').str[0].astype(int)
    
    # Calculate flooding areas for each category
    for category in [1, 2]:
        dep_cat = dep_flat[dep_flat['Flooding_Category'] <= category]
        area_col = f'dep_moderate_{category}_area'
        frac_col = f'dep_moderate_{category}_frac'
        
        attr_df[area_col] = (
            gpd.overlay(geo_gdf, dep_cat, how='intersection')
            .groupby(id_column)['geometry']
            .apply(lambda geom: geom.area.sum())
            .reindex(attr_df.index).fillna(0)
        )
        attr_df[frac_col] = attr_df[area_col] / attr_df['area']
    
    return attr_df


def load_clogged_cb_311(base_dir: Path, geo_gdf: gpd.GeoDataFrame,
                        attr_df: pd.DataFrame, id_column: str, logger) -> pd.DataFrame:
    """Load and aggregate clogged catch basin 311 complaints."""
    cb311_path = base_dir / 'aggregation' / 'flooding' / 'data' / 'nyc311_clogged_cb_jun_sep.csv'
    
    if not cb311_path.exists():
        logger.warning(f"Clogged CB 311 data not found at {cb311_path}")
        return attr_df
    
    clogged = pd.read_csv(cb311_path).dropna(subset=['latitude', 'longitude'])
    logger.info(f"Loaded {len(clogged)} clogged CB 311 complaints")
    
    # Parse dates and calculate days clogged
    clogged['created_date'] = pd.to_datetime(clogged['created_date'])
    clogged['closed_date'] = pd.to_datetime(clogged['closed_date'])
    end_date = pd.Timestamp('2023-09-29')
    clogged['effective_closed'] = clogged['closed_date'].fillna(end_date).clip(upper=end_date)
    clogged['days_clogged'] = (clogged['effective_closed'] - clogged['created_date']).dt.days.clip(lower=0)
    
    # Convert to GeoDataFrame
    clogged_geo = gpd.GeoDataFrame(
        clogged,
        geometry=gpd.points_from_xy(clogged['longitude'], clogged['latitude']),
        crs=WGS
    ).to_crs(PROJ)
    
    # Spatial join
    clogged_with_area = gpd.sjoin(
        clogged_geo, 
        geo_gdf[[id_column, 'geometry']], 
        how='left', 
        predicate='within'
    )
    
    # Aggregate by area
    cb_days = clogged_with_area.groupby(id_column)['days_clogged'].sum()
    attr_df['cb_days_clogged'] = cb_days.reindex(attr_df.index).fillna(0)
    
    # Average resolution time
    avg_resolution = clogged_with_area.groupby(id_column)['days_clogged'].mean()
    city_median = clogged['days_clogged'].median()
    attr_df['has_clogged_cb_complaint'] = avg_resolution.reindex(attr_df.index).notna().astype(int)
    attr_df['cb_avg_resolution_time'] = avg_resolution.reindex(attr_df.index).fillna(city_median)
    
    return attr_df


def load_topology(paths: GeometryPaths, attr_df: pd.DataFrame, logger) -> pd.DataFrame:
    """Load topology data if available."""
    topology_path = paths.topology_path
    
    if not topology_path.exists():
        logger.warning(f"Topology data not found at {topology_path}")
        return attr_df
    
    topology = pd.read_csv(topology_path, index_col=0)
    config = paths.config
    topology[config.id_column] = topology[config.id_column].astype(str)
    topology = topology.set_index(config.id_column)
    topology.columns = ['ft_elevation_' + c for c in topology.columns]
    
    attr_df = attr_df.merge(topology, left_index=True, right_index=True, how='left')
    logger.info(f"Merged topology data")
    
    return attr_df


def validate_and_clean(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Validate and clean the final dataframe."""
    # Check for NA values
    cols_allowed_na = ['empirical_estimate']
    for col in df.columns:
        if col in cols_allowed_na:
            continue
        na_count = df[col].isna().sum()
        if na_count > 0:
            logger.warning(f"Column {col} has {na_count} NA values")
    
    # Drop unwanted columns
    TO_DROP = ['tract_id', 'n_images_by_area_', 'CTLabel']
    current_cols = df.columns.tolist()
    for pattern in TO_DROP:
        df = df.loc[:, ~df.columns.str.contains(pattern)]
    
    dropped = set(current_cols) - set(df.columns)
    if dropped:
        logger.info(f"Dropped columns: {dropped}")
    
    return df


def aggregate_context_df(
    geometry_type: GeometryType,
    base_dir: Path = None,
    output_dir: Path = None,
    load_flooding: bool = True,
) -> pd.DataFrame:
    """
    Main aggregation function.
    
    Parameters
    ----------
    geometry_type : GeometryType
        Census geometry type to aggregate to
    base_dir : Path, optional
        Base directory for data files
    output_dir : Path, optional
        Output directory for results
    load_flooding : bool
        Whether to load flooding model data
        
    Returns
    -------
    pd.DataFrame
        Aggregated context dataframe
    """
    logger = setup_logger(f"aggregate-{geometry_type.value}")
    logger.setLevel("INFO")
    
    # Setup paths
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    paths = get_geometry_paths(geometry_type, str(base_dir))
    config = paths.config
    
    if output_dir is None:
        output_dir = paths.aggregation_dir
    
    logger.info(f"Aggregating data for {config.display_name}s")
    
    # Load base geometry
    attr_df, geo_gdf = load_geometry_data(paths, logger)
    
    # Add area column
    attr_df = attr_df.merge(
        geo_gdf[[config.id_column, 'area']], 
        left_index=True, 
        right_on=config.id_column
    ).set_index(config.id_column)
    
    # Load flooding model data if available and requested
    if load_flooding:
        attr_df = load_flooding_data(paths, attr_df, logger)
    
    # Load topology data
    attr_df = load_topology(paths, attr_df, logger)
    
    # Load and aggregate external datasets
    attr_df = load_floodnet_sensors(base_dir, geo_gdf, attr_df, config.id_column, logger)
    attr_df = load_catch_basins(base_dir, geo_gdf, attr_df, config.id_column, logger)
    attr_df = load_clogged_cb_311(base_dir, geo_gdf, attr_df, config.id_column, logger)
    attr_df = load_dep_stormwater(base_dir, geo_gdf, attr_df, config.id_column, logger)
    attr_df, geo_gdf = load_311_complaints(base_dir, geo_gdf, attr_df, config.id_column, logger)
    
    # Merge 311 columns from geo_gdf
    cols_311 = [c for c in geo_gdf.columns if '_311c' in c or c == 'n_311_reports']
    if cols_311:
        geo_311 = geo_gdf[[config.id_column] + cols_311].drop(columns=['geometry'], errors='ignore')
        attr_df = attr_df.merge(geo_311, left_index=True, right_on=config.id_column, how='left')
        if config.id_column in attr_df.columns:
            attr_df = attr_df.set_index(config.id_column)
    
    # Validate and clean
    attr_df = validate_and_clean(attr_df, logger)
    
    # Save outputs
    todays_date = datetime.now().strftime('%m%d%Y')
    output_path = output_dir / f'context_df_{geometry_type.value}_{todays_date}.csv'
    describe_path = output_dir / f'context_df_{geometry_type.value}_describe_{todays_date}.csv'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    attr_df.to_csv(output_path, index=True)
    attr_df.describe().apply(lambda s: s.apply(lambda x: format(x, 'f'))).to_csv(describe_path)
    
    logger.info(f"Saved context dataframe to {output_path}")
    logger.info(f"Saved description to {describe_path}")
    logger.success(f"Aggregation complete: {len(attr_df)} {config.display_name}s")
    
    return attr_df


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate flooding data to census geography level"
    )
    parser.add_argument(
        '--geometry-type', '-g',
        type=str,
        choices=['ct', 'cbg', 'cb'],
        default='ct',
        help='Census geometry type (ct=tract, cbg=block group, cb=block)'
    )
    parser.add_argument(
        '--no-flooding',
        action='store_true',
        help='Skip loading flooding model data'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory (default: aggregation/)'
    )
    
    args = parser.parse_args()
    
    geometry_type = GeometryType(args.geometry_type)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    aggregate_context_df(
        geometry_type=geometry_type,
        output_dir=output_dir,
        load_flooding=not args.no_flooding,
    )


if __name__ == '__main__':
    main()

