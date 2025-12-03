"""
Analysis DataFrame generation for bayflood pipeline.

This module generates comprehensive analysis DataFrames by combining ICAR model
estimates with various external data sources (demographics, flooding indicators, etc.).
Supports multiple census geometry types.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from glob import glob
import os
from typing import Optional, Dict, List, Union
import logging
from pathlib import Path

from shapely import wkt

from geometry_config import (
    GeometryType,
    GeometryPaths,
    get_geometry_paths,
    get_geometry_config,
)


def generate_nyc_analysis_df(
    run_dir: str,
    custom_prefix: str,
    base_dir: str = ".",
    logger: Optional[logging.Logger] = None,
    geometry_type: Union[GeometryType, str] = GeometryType.CT,
) -> pd.DataFrame:
    """
    Generate a comprehensive NYC census geography analysis DataFrame combining multiple data sources.
    
    Parameters:
    -----------
    run_dir : str
        Directory containing ICAR estimate files
    custom_prefix : str
        Prefix for output files
    base_dir : str, optional
        Base directory for all data files, defaults to current directory
    logger : logging.Logger, optional
        Logger instance for tracking progress
    geometry_type : GeometryType or str, optional
        Census geometry type (ct, cbg, cb). Defaults to ct (Census Tracts).
    
    Returns:
    --------
    pd.DataFrame
        Combined analysis DataFrame with all metrics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
    # Handle geometry type
    if isinstance(geometry_type, str):
        geometry_type = GeometryType(geometry_type.lower())
    
    paths = get_geometry_paths(geometry_type, base_dir)
    config = paths.config
    id_column = config.id_column

    # Constants
    WGS = 'EPSG:4326'
    PROJ = 'EPSG:2263'
    
    logger.info(f"Generating analysis DataFrame for {config.display_name}s")
    
    def parse_md(md: pd.DataFrame) -> pd.DataFrame:
        """Parse ACS metadata."""
        md = pd.json_normalize(md['variables']).set_index(md.index)
        min_sep = min(md['label'].apply(lambda x: x.count('!!')))
        max_sep = max(md['label'].apply(lambda x: x.count('!!')))
        
        for i in range(min_sep + 1, max_sep + 2):
            md[f'desc_{i}'] = md['label'].apply(
                lambda x: x.split('!!')[i-1] if len(x.split('!!')) >= i else None
            )
        
        TO_DROP = ['label', 'concept', 'predicateType', 'group', 'limit', 'predicateOnly']
        md = md.drop(columns=TO_DROP)
        md = md[md['desc_1'].isin(['Estimate'])].sort_index()
        return md

    def parse_acs(acs: pd.DataFrame, cols: Dict[str, str]) -> pd.DataFrame:
        """Parse ACS data."""
        acs.columns = acs.iloc[0]
        acs = acs[1:]
        # Use generic geoid column name
        acs['geoid'] = acs['GEO_ID'].str.split('US', expand=True)[1]
        acs = acs.set_index('geoid')
        acs = acs[list(cols.keys())]
        acs.columns = acs.columns.map(lambda x: cols[x])
        return acs.astype(int)

    # Load ICAR estimates
    icar_files = glob(f"{run_dir}/estimate*.csv")
    logger.info(f"Found {len(icar_files)} ICAR estimates.")
    
    icar_estimates = {}
    for f in icar_files:
        df = pd.read_csv(f)
        # Handle both tract_id and geoid column names
        id_col = 'geoid' if 'geoid' in df.columns else 'tract_id'
        df[id_col] = df[id_col].astype(int).astype(str)
        df['geoid'] = df[id_col]  # Normalize to geoid
        icar_estimates[os.path.splitext(os.path.basename(f))[0]] = df

    # Load census geometry
    geojson_path = paths.aggregation_geojson_path
    if not geojson_path.exists():
        geojson_path = paths.geojson_path
        
    geo_df = gpd.read_file(str(geojson_path))
    length_before_processing = len(geo_df.index)
    
    # Standard columns to drop
    TO_DROP = ['OBJECTID', 'BoroCode', 'CT2020', 'CDEligibil', 'NTA2020', 
               'CDTA2020', 'Shape__Area', 'Shape__Length', 'geometry']
    cols_to_drop = [c for c in TO_DROP if c in geo_df.columns]
    geo_df = geo_df.drop(columns=cols_to_drop)
    geo_df = geo_df.set_index(id_column).astype(str)
    
    logger.info(f"Loaded {len(geo_df)} {config.display_name}s")

    # Load positive images (if available)
    sep29_positives_path = Path(base_dir) / 'data' / 'processed' / 'sep29_positives.csv'
    if sep29_positives_path.exists():
        sep29_positives = pd.read_csv(sep29_positives_path)
        sep29_positives = sep29_positives[['image_path', 'sentiment_1']]
        sep29_positives = sep29_positives.rename(columns={'sentiment_1': 'positive_image', 'image_path': 'image_path'})
        sep29_positives['frame_id'] = sep29_positives['image_path'].str.split('/').str[-1].str.split('.').str[0]

        # Load sep29 metadata 
        sep29_md_path = Path(base_dir) / 'data' / 'md.csv'
        if sep29_md_path.exists():
            sep29_md = pd.read_csv(sep29_md_path)
            sep29_md_positives = sep29_md[sep29_md['frame_id'].isin(sep29_positives['frame_id'])]
            sep29_md_positives = gpd.GeoDataFrame(
                sep29_md_positives, 
                geometry=gpd.points_from_xy(
                    sep29_md_positives['gps_info.longitude'], 
                    sep29_md_positives['gps_info.latitude']
                ), 
                crs=WGS
            ).to_crs(PROJ)

    # Merge ICAR estimates
    if 'estimate_p_y' in icar_estimates:
        geo_df = geo_df.merge(
            icar_estimates['estimate_p_y'], 
            left_index=True, 
            right_on='geoid', 
            suffixes=('', '_p_y')
        ).set_index('geoid')
        
    if 'estimate_at_least_one_positive_image_by_area' in icar_estimates:
        geo_df = geo_df.merge(
            icar_estimates['estimate_at_least_one_positive_image_by_area'], 
            left_index=True, 
            right_on='geoid', 
            suffixes=('', '_p_alop')
        ).set_index('geoid')

    # Load and merge demographic data (CT-level only for now)
    # TODO: Add support for CBG and CB level ACS data
    if geometry_type == GeometryType.CT:
        # Race data
        race_cols = {
            'DP05_0001E': 'total_population',
            'DP05_0079E': 'nhl_white_alone', 
            'DP05_0080E': 'nhl_black_alone', 
            'DP05_0073E': 'hispanic_alone', 
            'DP05_0082E': 'nhl_asian_alone',
            'DP05_0019E': 'n_children', 
            'DP05_0024E': 'n_elderly',
        }
        
        dp05_path = Path(base_dir) / 'aggregation' / 'demo' / 'data' / 'acs2023_dp05.json'
        if dp05_path.exists():
            dp05_nyc = pd.read_json(dp05_path, orient='records')
            race_nyc = parse_acs(dp05_nyc, race_cols)
            geo_df = geo_df.merge(race_nyc, left_index=True, right_index=True, how='left')

        # Internet access data
        internet_cols = {
            'S2801_C01_001E': 'total_households',
            'S2801_C01_012E': 'num_households_with_internet',
            'S2801_C01_005E': 'num_households_with_smartphone'
        }
        s2801_path = Path(base_dir) / 'aggregation' / 'demo' / 'data' / 'acs2023_s2801.json'
        if s2801_path.exists():
            s2801_nyc = pd.read_json(s2801_path, orient='records')
            internet_nyc = parse_acs(s2801_nyc, internet_cols)
            geo_df = geo_df.join(internet_nyc)

        # Income data
        income_cols = {'S1901_C01_012E': 'median_household_income'}
        s1901_path = Path(base_dir) / 'aggregation' / 'demo' / 'data' / 'acs2023_s1901.json'
        if s1901_path.exists():
            s1901_nyc = pd.read_json(s1901_path, orient='records')
            income_nyc = parse_acs(s1901_nyc, income_cols)
            geo_df = geo_df.join(income_nyc)

        # Education data
        education_cols = {
            'S1501_C01_009E': 'num_high_school_graduates',
            'S1501_C01_012E': 'num_bachelors_degree',
            'S1501_C01_013E': 'num_graduate_degree'
        }
        s1501_path = Path(base_dir) / 'aggregation' / 'demo' / 'data' / 'acs2023_s1501.json'
        if s1501_path.exists():
            s1501_nyc = pd.read_json(s1501_path, orient='records')
            education_nyc = parse_acs(s1501_nyc, education_cols)
            geo_df = geo_df.join(education_nyc)

        # Limited English speaking households
        leh_cols = {'S1602_C03_001E': 'num_limited_english_speaking_households'}
        s1602_path = Path(base_dir) / 'aggregation' / 'demo' / 'data' / 'acs2023_s1602.json'
        if s1602_path.exists():
            s1602_nyc = pd.read_json(s1602_path, orient='records')
            leh_nyc = parse_acs(s1602_nyc, leh_cols)
            geo_df = geo_df.join(leh_nyc)

    # Topology data
    topology_path = paths.topology_path
    if topology_path.exists():
        topology_df = pd.read_csv(topology_path, index_col=0)
        topology_df[id_column] = topology_df[id_column].astype(str)
        topology_df = topology_df.set_index(id_column)
        topology_df.columns = ['ft_elevation_' + c for c in topology_df.columns]
        geo_df = geo_df.merge(topology_df, left_index=True, right_index=True, how='left')

    # Add geographic features
    gdf_geo = gpd.read_file(str(geojson_path)).to_crs(PROJ)[[id_column, 'geometry']]
    gdf_geo['area'] = gdf_geo.area
    geo_df = geo_df.merge(gdf_geo[[id_column, 'area']], left_index=True, right_on=id_column).set_index(id_column)

    # Add FloodNet sensor data
    floodnet_path = Path(base_dir) / 'aggregation' / 'flooding' / 'static' / 'current_floodnet_sensors.csv'
    if floodnet_path.exists():
        dec1224_floodnet_sensors = pd.read_csv(floodnet_path, engine='pyarrow')
        all_floodnet_sensors_geo = dec1224_floodnet_sensors.groupby('deployment_id').first()
        
        logger.info(f"Found {len(all_floodnet_sensors_geo)} sensors in the December 2024 dataset.")

        distance_thres = 1 
        all_floodnet_sensors_geo = gpd.GeoDataFrame(
            all_floodnet_sensors_geo, 
            geometry=gpd.points_from_xy(all_floodnet_sensors_geo['longitude'], all_floodnet_sensors_geo['latitude']), 
            crs='EPSG:4326'
        ).to_crs(2263)

        # Drop sensors within distance_thres of each other
        to_drop = []
        for i, row in all_floodnet_sensors_geo.iterrows():
            if i in to_drop:
                continue
            for j, row2 in all_floodnet_sensors_geo.iterrows():
                if i == j:
                    continue
                if row.geometry.distance(row2.geometry) < distance_thres:
                    to_drop.append(j)
        all_floodnet_sensors_geo = all_floodnet_sensors_geo.drop(to_drop)
        logger.info(f"Loaded {len(all_floodnet_sensors_geo)} unique FloodNet sensors.")

        geo_df['n_floodnet_sensors'] = gpd.sjoin(gdf_geo, all_floodnet_sensors_geo).groupby(id_column).size().reindex(geo_df.index).fillna(0)

    # Add catch basins data
    catch_basins_path = Path(base_dir) / 'aggregation' / 'flooding' / 'static' / 'catch_basins_nyc.geojson'
    if catch_basins_path.exists():
        catch_basins = gpd.read_file(catch_basins_path).to_crs(PROJ)
        logger.info(f"Loaded {len(catch_basins)} catch basins.")
        
        geo_df['n_catch_basins'] = gpd.sjoin(gdf_geo, catch_basins).groupby(id_column).size().reindex(geo_df.index).fillna(0)
        geo_df['catch_basin_density'] = geo_df['n_catch_basins'] / geo_df['area']
        logger.info(f"Total catch basins: {geo_df['n_catch_basins'].sum()}")

    # Add catch basin days clogged (from 311 complaints June-Sep 2023)
    clogged_cb_path = Path(base_dir) / 'aggregation' / 'flooding' / 'data' / 'nyc311_clogged_cb_jun_sep.csv'
    if clogged_cb_path.exists():
        clogged_cb_311 = pd.read_csv(clogged_cb_path)
        clogged_cb_311 = clogged_cb_311.dropna(subset=['latitude', 'longitude'])
        logger.info(f"Loaded {len(clogged_cb_311)} clogged catch basin 311 complaints.")
        
        clogged_cb_311['created_date'] = pd.to_datetime(clogged_cb_311['created_date'])
        clogged_cb_311['closed_date'] = pd.to_datetime(clogged_cb_311['closed_date'])
        end_date = pd.Timestamp('2023-09-29')
        clogged_cb_311['effective_closed'] = clogged_cb_311['closed_date'].fillna(end_date).clip(upper=end_date)
        clogged_cb_311['days_clogged'] = (clogged_cb_311['effective_closed'] - clogged_cb_311['created_date']).dt.days.clip(lower=0)
        
        clogged_cb_311_geo = gpd.GeoDataFrame(
            clogged_cb_311,
            geometry=gpd.points_from_xy(clogged_cb_311['longitude'], clogged_cb_311['latitude']),
            crs=WGS
        ).to_crs(PROJ)
        
        clogged_with_geo = gpd.sjoin(clogged_cb_311_geo, gdf_geo[[id_column, 'geometry']], how='left', predicate='within')
        cb_days_clogged = clogged_with_geo.groupby(id_column)['days_clogged'].sum()
        geo_df['cb_days_clogged'] = cb_days_clogged.reindex(geo_df.index).fillna(0)
        
        avg_resolution = clogged_with_geo.groupby(id_column)['days_clogged'].mean()
        city_median_resolution = clogged_cb_311['days_clogged'].median()
        geo_df['has_clogged_cb_complaint'] = avg_resolution.reindex(geo_df.index).notna().astype(int)
        geo_df['cb_avg_resolution_time'] = avg_resolution.reindex(geo_df.index).fillna(city_median_resolution)

    # Add DEP stormwater data
    dep_path = Path(base_dir) / 'aggregation' / 'flooding' / 'static' / 'dep_stormwater_moderate_current' / 'data.gdb'
    if dep_path.exists():
        dep_moderate = gpd.read_file(str(dep_path)).to_crs(PROJ)
        
        polygons = {}
        for i, row in dep_moderate.iterrows():
            for idx, polygon in enumerate(row['geometry'].geoms):
                polygons[f'{row["Flooding_Category"]}_{idx}'] = polygon
        
        dep_moderate_flat = gpd.GeoDataFrame(polygons, index=['geometry']).T
        dep_moderate_flat.set_geometry('geometry', inplace=True)
        dep_moderate_flat.crs = dep_moderate.crs
        dep_moderate_flat['Flooding_Category'] = dep_moderate_flat.index.str.split('_').str[0].astype(int)

        for category in [1, 2]:
            dep_cat = dep_moderate_flat[dep_moderate_flat['Flooding_Category'] <= category]
            area_col = f'dep_moderate_{category}_area'
            frac_col = f'dep_moderate_{category}_frac'
            
            geo_df[area_col] = gpd.overlay(gdf_geo, dep_cat, how='intersection').groupby(id_column)['geometry'].apply(
                lambda geom: geom.area.sum()
            ).reindex(geo_df.index).fillna(0)
            geo_df[frac_col] = geo_df[area_col] / geo_df['area']

    # Add 311 data
    nyc311_path = Path(base_dir) / 'aggregation' / 'flooding' / 'data' / 'nyc311_flooding_sep29.csv'
    if nyc311_path.exists():
        nyc311_sep29 = pd.read_csv(nyc311_path).dropna(subset=['latitude', 'longitude'])
        nyc311_sep29 = gpd.GeoDataFrame(
            nyc311_sep29, 
            geometry=gpd.points_from_xy(nyc311_sep29.longitude, nyc311_sep29.latitude),
            crs=WGS
        ).to_crs(PROJ)

        for descriptor in nyc311_sep29['descriptor'].unique():
            desc = descriptor.split('(')[0].strip().lower().replace(' ', '_') + '_311c'
            gdf_geo[desc] = gdf_geo['geometry'].apply(
                lambda x: nyc311_sep29[nyc311_sep29['descriptor'] == descriptor].within(x).sum()
            )

    # Validation
    COLS_ALLOWED_NA_VALS = ['empirical_estimate']
    for c in geo_df.columns:
        if c in COLS_ALLOWED_NA_VALS:
            continue
        if geo_df[c].isna().sum() > 0:
            logger.warning(f"Column {c} has {geo_df[c].isna().sum()} NA values.")

    # Final cleaning
    cols_to_drop = ['area', 'geometry']
    gdf_geo = gdf_geo.drop(columns=[c for c in cols_to_drop if c in gdf_geo.columns])
    geo_df = geo_df.merge(gdf_geo, left_index=True, right_on=id_column, how='left')
    
    TO_DROP = ['tract_id', 'geoid', 'n_images_by_area_', 'CTLabel']
    current_cols = geo_df.columns
    for pattern in TO_DROP:
        geo_df = geo_df.loc[:, ~geo_df.columns.str.contains(pattern)]
    
    dropped = set(current_cols) - set(geo_df.columns)
    if dropped:
        logger.info(f"Dropped columns: {dropped}")

    # Note: We don't assert row count for CBG/CB since merging estimates may not cover all areas
    if geometry_type == GeometryType.CT:
        assert length_before_processing == len(geo_df.index), "Number of rows changed during processing."

    # Save outputs
    todays_date = pd.to_datetime('today').strftime('%m%d%Y')
    geo_df.describe().apply(lambda s: s.apply(lambda x: format(x, 'f'))).to_csv(
        f'{run_dir}/analysis_df_describe_{custom_prefix}_{todays_date}.csv'
    )
    geo_df.to_csv(f'{run_dir}/analysis_df_{custom_prefix}_{todays_date}.csv', index=False)

    logger.info(f"Saved analysis DataFrame with {len(geo_df)} {config.display_name}s")
    
    return geo_df
