import pandas as pd
import geopandas as gpd
import numpy as np
from glob import glob
import os
from typing import Optional, Dict, List
import logging

def generate_nyc_analysis_df(
    run_dir: str,
    custom_prefix: str,
    use_smoothing: bool,
    base_dir: str = ".",
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Generate a comprehensive NYC census tract analysis DataFrame combining multiple data sources.
    
    Parameters:
    -----------
    run_dir : str
        Directory containing ICAR estimate files
    custom_prefix : str
        Prefix for output files
    use_smoothing : bool
        Whether to use smoothing in the analysis
    base_dir : str, optional
        Base directory for all data files, defaults to current directory
    logger : logging.Logger, optional
        Logger instance for tracking progress
    
    Returns:
    --------
    pd.DataFrame
        Combined analysis DataFrame with all metrics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    # Constants
    WGS = 'EPSG:4326'
    PROJ = 'EPSG:2263'
    
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
        acs['tract_id'] = acs['GEO_ID'].str.split('US', expand=True)[1]
        acs = acs.set_index('tract_id')
        acs = acs[list(cols.keys())]
        acs.columns = acs.columns.map(lambda x: cols[x])
        return acs.astype(int)

    # Load ICAR estimates
    icar_files = glob(f"{run_dir}/estimate*.csv")
    logger.info(f"Found {len(icar_files)} ICAR estimates.")
    
    icar_estimates = {}
    for f in icar_files:
        df = pd.read_csv(f)
        df['tract_id'] = df['tract_id'].astype(int).astype(str)
        icar_estimates[os.path.splitext(os.path.basename(f))[0]] = df

    # Load census tract geometries
    ct_nyc = gpd.read_file(f'{base_dir}/aggregation/geo/data/ct-nyc-wi-2020.geojson')
    TO_DROP = ['OBJECTID', 'BoroCode', 'CT2020', 'CDEligibil', 'NTA2020', 'CDTA2020', 'Shape__Area', 'Shape__Length', 'geometry']
    ct_nyc.drop(columns=TO_DROP, inplace=True)
    ct_nyc = ct_nyc.set_index('GEOID').astype(str)

    # Merge ICAR estimates
    ct_nyc = ct_nyc.merge(icar_estimates['estimate_p_y'], left_index=True, right_on='tract_id', suffixes=('_ct', '_p_y')).set_index('tract_id')
    ct_nyc = ct_nyc.merge(icar_estimates['estimate_at_least_one_positive_image_by_area'], left_index=True, right_on='tract_id', suffixes=('_ct', '_p_alop')).set_index('tract_id')
    ct_nyc = ct_nyc.merge(icar_estimates['estimate_at_least_one_positive_image_by_area_if_you_have_100_images'], left_index=True, right_on='tract_id', suffixes=('_ct', '_p_y_100')).set_index('tract_id')

    # Load and merge demographic data
    # Race data
    race_cols = {
        'DP05_0001E': 'total_population',
        'DP05_0079E': 'nhl_white_alone',
        'DP05_0080E': 'nhl_black_alone',
        'DP05_0073E': 'hispanic_alone',
        'DP05_0082E': 'nhl_asian_alone'
    }
    dp05_nyc = pd.read_json(f'{base_dir}/aggregation/demo/data/acs22_dp05.json', orient='records')
    race_nyc = parse_acs(dp05_nyc, race_cols)
    ct_nyc = ct_nyc.merge(race_nyc, left_index=True, right_index=True)

    # Internet access data
    internet_cols = {
        'S2801_C01_001E': 'total_households',
        'S2801_C01_012E': 'num_households_with_internet',
        'S2801_C01_005E': 'num_households_with_smartphone'
    }
    s2801_nyc = pd.read_json(f'{base_dir}/aggregation/demo/data/acs22_s2801.json', orient='records')
    internet_nyc = parse_acs(s2801_nyc, internet_cols)
    ct_nyc = ct_nyc.join(internet_nyc)

    # Income data
    income_cols = {'S1901_C01_012E': 'median_household_income'}
    s1901_nyc = pd.read_json(f'{base_dir}/aggregation/demo/data/acs22_s1901.json', orient='records')
    income_nyc = parse_acs(s1901_nyc, income_cols)
    ct_nyc = ct_nyc.join(income_nyc)

    # Education data
    education_cols = {
        'S1501_C01_009E': 'num_high_school_graduates',
        'S1501_C01_012E': 'num_bachelors_degree',
        'S1501_C01_013E': 'num_graduate_degree'
    }
    s1501_nyc = pd.read_json(f'{base_dir}/aggregation/demo/data/acs22_s1501.json', orient='records')
    education_nyc = parse_acs(s1501_nyc, education_cols)
    ct_nyc = ct_nyc.join(education_nyc)

    # Limited English speaking households
    leh_cols = {'S1602_C03_001E': 'num_limited_english_speaking_households'}
    s1602_nyc = pd.read_json(f'{base_dir}/aggregation/demo/data/acs22_s1602.json', orient='records')
    leh_nyc = parse_acs(s1602_nyc, leh_cols)
    ct_nyc = ct_nyc.join(leh_nyc)

    # Topology data
    topology_ct_nyc = pd.read_csv(f'{base_dir}/aggregation/geo/data/processed/ct_nyc_topology.csv', index_col=0)
    topology_ct_nyc['GEOID'] = topology_ct_nyc['GEOID'].astype(str)
    topology_ct_nyc = topology_ct_nyc.set_index('GEOID')
    topology_ct_nyc.columns = ['ft_elevation_' + c for c in topology_ct_nyc.columns]
    ct_nyc = ct_nyc.merge(topology_ct_nyc, left_index=True, right_on='GEOID')

    # Add geographic features
    gdf_ct_nyc = gpd.read_file(f'{base_dir}/aggregation/geo/data/ct-nyc-wi-2020.geojson').to_crs(PROJ)[['GEOID', 'geometry']]
    gdf_ct_nyc['area'] = gdf_ct_nyc.area
    ct_nyc = ct_nyc.merge(gdf_ct_nyc[['GEOID', 'area']], left_index=True, right_on='GEOID').set_index('GEOID')

    # Add FloodNet sensor data
    floodnet_sensor = pd.read_csv(f'{base_dir}/aggregation/flooding/static/floodnet-flood-sensor-sep-2023.csv', engine='pyarrow')
    floodnet_tide = pd.read_csv(f'{base_dir}/aggregation/flooding/static/floodnet-tide-sep-2023.csv', engine='pyarrow')
    floodnet_weather = pd.read_csv(f'{base_dir}/aggregation/flooding/static/floodnet-weather-sep-2023.csv', engine='pyarrow')

    all_sensors = pd.concat([
        floodnet_sensor.groupby('deployment_id').first()[['lat', 'lon']].reset_index(),
        floodnet_tide.groupby('sensor_id').first()[['lat', 'lon']].reset_index(),
        floodnet_weather.groupby('sensor_id').first()[['lat', 'lon']].reset_index()
    ])
    
    all_sensors_geo = gpd.GeoDataFrame(
        all_sensors, 
        geometry=gpd.points_from_xy(all_sensors.lon, all_sensors.lat),
        crs=WGS
    ).to_crs(PROJ)

    ct_nyc['n_floodnet_sensors'] = gpd.sjoin(gdf_ct_nyc, all_sensors_geo).groupby('GEOID').size().reindex(ct_nyc.index).fillna(0)

    # Add DEP stormwater data
    dep_moderate = gpd.read_file(f'{base_dir}/aggregation/flooding/static/dep_stormwater_moderate_current/data.gdb').to_crs(PROJ)
    
    # Flatten DEP moderate flooding polygons
    polygons = {}
    for i, row in dep_moderate.iterrows():
        for idx, polygon in enumerate(row['geometry'].geoms):
            polygons[f'{row["Flooding_Category"]}_{idx}'] = polygon
    
    dep_moderate_flat = gpd.GeoDataFrame(polygons, index=['geometry']).T
    dep_moderate_flat.set_geometry('geometry', inplace=True)
    dep_moderate_flat.crs = dep_moderate.crs
    dep_moderate_flat['Flooding_Category'] = dep_moderate_flat.index.str.split('_').str[0].astype(int)

    # Calculate flooding areas
    for category in [1, 2]:
        dep_cat = dep_moderate_flat[dep_moderate_flat['Flooding_Category'] == category]
        area_col = f'dep_moderate_{category}_area'
        frac_col = f'dep_moderate_{category}_frac'
        
        ct_nyc[area_col] = gpd.overlay(gdf_ct_nyc, dep_cat, how='intersection').groupby('GEOID')['geometry'].apply(
            lambda geom: geom.area.sum()
        ).reindex(ct_nyc.index).fillna(0)
        
        ct_nyc[frac_col] = ct_nyc[area_col] / ct_nyc['area']

    # Add 311 data
    nyc311_sep29 = pd.read_csv(f'{base_dir}/aggregation/flooding/data/nyc311_flooding_sep29.csv').dropna(subset=['latitude', 'longitude'])
    nyc311_sep29 = gpd.GeoDataFrame(
        nyc311_sep29, 
        geometry=gpd.points_from_xy(nyc311_sep29.longitude, nyc311_sep29.latitude),
        crs=WGS
    ).to_crs(PROJ)

    for descriptor in nyc311_sep29['descriptor'].unique():
        desc = descriptor.split('(')[0].strip().lower().replace(' ', '_') + '_311c'
        gdf_ct_nyc[desc] = gdf_ct_nyc['geometry'].apply(
            lambda x: nyc311_sep29[nyc311_sep29['descriptor'] == descriptor].within(x).sum()
        )

    # Add NYC Flood Vulnerability Index
    nyc_fvi = pd.read_csv(f'{base_dir}/aggregation/flooding/data/nyc_fvi.csv')
    nyc_fvi['geoid'] = nyc_fvi['geoid'].astype(str)
    ct_nyc = ct_nyc.join(nyc_fvi.set_index('geoid'))

    # Final cleaning
    gdf_ct_nyc.drop(columns=['area', 'geometry'], inplace=True)
    ct_nyc = ct_nyc.merge(gdf_ct_nyc, left_index=True, right_on='GEOID')
    
    TO_DROP = ['tract_id', 'n_images_by_area_', 'CTLabel']
    current_cols = ct_nyc.columns
    ct_nyc = ct_nyc.loc[:, ~ct_nyc.columns.str.contains('|'.join(TO_DROP))]
    
    logger.info(f"Dropped columns: {set(current_cols) - set(ct_nyc.columns)}")

    # Save outputs
    todays_date = pd.to_datetime('today').strftime('%m%d%Y')
    ct_nyc.describe().apply(lambda s: s.apply(lambda x: format(x, 'f'))).to_csv(
        f'{run_dir}/analysis_df_describe_{custom_prefix}_{todays_date}.csv'
    )
    ct_nyc.to_csv(f'{run_dir}/analysis_df_{custom_prefix}_{todays_date}.csv', index=False)

    return ct_nyc