import pandas as pd
import geopandas as gpd
import numpy as np
from glob import glob
import os
from typing import Optional, Dict, List
import logging

from shapely import wkt

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
    ct_nyc = gpd.read_file(f'{base_dir}/aggregation/geo/data/ct-nyc-2020.geojson')
    length_before_processing = len(ct_nyc.index)
    TO_DROP = ['OBJECTID', 'BoroCode', 'CT2020', 'CDEligibil', 'NTA2020', 'CDTA2020', 'Shape__Area', 'Shape__Length', 'geometry']
    ct_nyc.drop(columns=TO_DROP, inplace=True)
    ct_nyc = ct_nyc.set_index('GEOID').astype(str)

    # Load positive images 
    sep29_positives = pd.read_csv(f'{base_dir}/data/processed/sep29_positives.csv')
    sep29_positives = sep29_positives[['image_path', 'sentiment_1']]

    sep29_positives = sep29_positives.rename(columns={'sentiment_1': 'positive_image', 'image_path': 'image_path'})
    sep29_positives['frame_id'] = sep29_positives['image_path'].str.split('/').str[-1].str.split('.').str[0]

    # Load sep29 metadata 
    sep29_md = pd.read_csv(f'{base_dir}/data/md.csv')
    # Filter metadata to only include sep29 images and merge 
    sep29_md_positives = sep29_md[sep29_md['frame_id'].isin(sep29_positives['frame_id'])]

    sep29_md_positives = gpd.GeoDataFrame(sep29_md_positives, geometry=gpd.points_from_xy(sep29_md_positives['gps_info.longitude'], sep29_md_positives['gps_info.latitude']), crs=WGS).to_crs(PROJ)


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
    'DP05_0082E': 'nhl_asian_alone',
    'DP05_0019E': 'n_children', 
    'DP05_0024E': 'n_elderly',
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
    gdf_ct_nyc = gpd.read_file(f'{base_dir}/aggregation/geo/data/ct-nyc-2020.geojson').to_crs(PROJ)[['GEOID', 'geometry']]
    gdf_ct_nyc['area'] = gdf_ct_nyc.area
    ct_nyc = ct_nyc.merge(gdf_ct_nyc[['GEOID', 'area']], left_index=True, right_on='GEOID').set_index('GEOID')

    # Add FloodNet sensor data
    # FLOODNET 
    sep2923_floodnet_sensors = pd.read_csv(f'{base_dir}/aggregation/flooding/static/floodnet-flood-sensor-sep-2023.csv', engine='pyarrow')

    dec1224_floodnet_sensors = pd.read_csv(f'{base_dir}/aggregation/flooding/static/current_floodnet_sensors.csv', engine='pyarrow')


    sep2923_floodnet_sensors_geo = sep2923_floodnet_sensors.groupby('deployment_id').first()
    all_floodnet_sensors_geo = dec1224_floodnet_sensors.groupby('deployment_id').first()

    logger.info(f"Found {len(sep2923_floodnet_sensors_geo)} sensors in the September 2023 dataset.")
    logger.info(f"Found {len(all_floodnet_sensors_geo)} sensors in the December 2024 dataset.")

    distance_thres=1 

    all_floodnet_sensors_geo = gpd.GeoDataFrame(all_floodnet_sensors_geo, geometry=gpd.points_from_xy(all_floodnet_sensors_geo['longitude'], all_floodnet_sensors_geo['latitude']), crs='EPSG:4326').to_crs(2263)

    # drop sensors that are within distance_thres of each other
    # for each sensor, check if there is a sensor within distance_thres
    # if so, drop it
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
    logger.info(f"Dropped {len(to_drop)} sensors that were within {distance_thres} foot of each other, so that there are {len(all_floodnet_sensors_geo)} unique sensors.")


    logger.info("Loaded and processed Floodnet sensor data.")

    del sep2923_floodnet_sensors, dec1224_floodnet_sensors

    ct_nyc['n_floodnet_sensors'] = gpd.sjoin(gdf_ct_nyc, all_floodnet_sensors_geo).groupby('GEOID').size().reindex(ct_nyc.index).fillna(0)

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
        dep_cat = dep_moderate_flat[dep_moderate_flat['Flooding_Category'] <= category]
        area_col = f'dep_moderate_{category}_area'
        frac_col = f'dep_moderate_{category}_frac'
        
        ct_nyc[area_col] = gpd.overlay(gdf_ct_nyc, dep_cat, how='intersection').groupby('GEOID')['geometry'].apply(
            lambda geom: geom.area.sum()
        ).reindex(ct_nyc.index).fillna(0)
        
        
        ct_nyc[frac_col] = ct_nyc[area_col] / ct_nyc['area']


        # NEW JAN 27: dep flooding specifically around classified positive images 
        # Create 250ft buffers around all positive images at once
        #positive_buffers = sep29_md_positives.geometry.buffer(250)
        #positive_buffers_gdf = gpd.GeoDataFrame(geometry=positive_buffers, crs=PROJ)

        # Calculate flooded areas for each category
        #for category in [1, 2]:
        #    dep_cat = dep_moderate_flat[dep_moderate_flat['Flooding_Category'] == category]
        #    area_col = f'dep_moderate_{category}_area_250ft'
            
            # Vectorized intersection between all buffers and flood areas
        #    flooded_areas = gpd.overlay(positive_buffers_gdf, dep_cat, how='intersection')
        #    flooded_areas['area'] = flooded_areas.geometry.area
            
            # Assign areas back to original points
        #    sep29_md_positives[f'flooded_area_{category}_250ft'] = flooded_areas.groupby(level=0)['area'].sum()

            # log the distribution of flooded_area_250ft before grouping by tract 
        #    logger.info(f"Category {category} flooded area 250ft distribution, image level: {sep29_md_positives[f'flooded_area_{category}_250ft'].describe()}")
            
            # Spatial join to get tract-level summaries
        #    tract_floods = gpd.sjoin(
        #        gdf_ct_nyc, 
        #        gpd.GeoDataFrame(
        #            sep29_md_positives[['geometry', f'flooded_area_{category}_250ft']], 
        #            crs=PROJ
        #        ),
        #        predicate='contains',
        #        how='left'
        #    )
            
        #    flooded_area_250ft = tract_floods.groupby('GEOID')[f'flooded_area_{category}_250ft'].sum()
            # log the distribution of flooded_area_250ft 
        #    logger.info(f"Category {category} flooded area 250ft distribution, tract level: {flooded_area_250ft.describe()}")
            # cast GEOID to string 
        #    flooded_area_250ft.index = flooded_area_250ft.index.astype(str)

        #    ct_nyc = ct_nyc.merge(
        #        flooded_area_250ft.to_frame(), 
        #        left_index=True, 
        #        right_index=True, 
        #        how='left'
        #    ).fillna(0)

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
    #print(len(ct_nyc.index))
    #nyc_fvi = pd.read_csv(f'{base_dir}/aggregation/flooding/data/nyc_fvi.csv')
    #def try_geo_parse(x):
    #    try:
    #        return wkt.loads(x)
    #    except:
    #        return None
        
    #nyc_fvi['geometry'] = nyc_fvi['the_geom'].apply(try_geo_parse)
    # drop na geometry 
    #nyc_fvi = nyc_fvi.dropna(subset=['geometry'])
    # drop the_geom 
    #nyc_fvi.drop(columns=['the_geom'], inplace=True)
    #nyc_fvi = gpd.GeoDataFrame(nyc_fvi, crs=WGS, geometry='geometry').to_crs(PROJ)
    #nyc_fvi['geoid'] = nyc_fvi['geoid'].astype(str)
    #nyc_fvi = ct_nyc.merge(nyc_fvi.set_index('geoid'), left_index=True, right_index=True)[['fshri']]
    #print(len(nyc_fvi.index))
    #ct_nyc = ct_nyc.merge(nyc_fvi, left_index=True, right_index=True, how='left').drop_duplicates()
    #ct_nyc['fshri'] = ct_nyc['fshri'].fillna(0)
    #print(len(ct_nyc.index))

    COLS_ALLOWED_NA_VALS = ['empirical_estimate']
    def na_validation(df, cols_allowed_na_vals):
        for c in df.columns:
            if c in cols_allowed_na_vals:
                continue
            if df[c].isna().sum() > 0:
                logger.error(f"Column {c} has {df[c].isna().sum()} NA values.")
        else: 
            logger.success("No N/A values found in columns.")
    na_validation(ct_nyc, COLS_ALLOWED_NA_VALS)

    # Final cleaning
    gdf_ct_nyc.drop(columns=['area', 'geometry'], inplace=True)
    ct_nyc = ct_nyc.merge(gdf_ct_nyc, left_index=True, right_on='GEOID')
    
    TO_DROP = ['tract_id', 'n_images_by_area_', 'CTLabel']
    current_cols = ct_nyc.columns
    ct_nyc = ct_nyc.loc[:, ~ct_nyc.columns.str.contains('|'.join(TO_DROP))]
    
    logger.info(f"Dropped columns: {set(current_cols) - set(ct_nyc.columns)}")

    assert length_before_processing == len(ct_nyc.index), "Number of rows changed during processing."

    # Save outputs
    todays_date = pd.to_datetime('today').strftime('%m%d%Y')
    ct_nyc.describe().apply(lambda s: s.apply(lambda x: format(x, 'f'))).to_csv(
        f'{run_dir}/analysis_df_describe_{custom_prefix}_{todays_date}.csv'
    )
    ct_nyc.to_csv(f'{run_dir}/analysis_df_{custom_prefix}_{todays_date}.csv', index=False)

    return ct_nyc