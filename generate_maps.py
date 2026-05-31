"""
Map generation module for bayflood pipeline.

Generates visualization maps for flooding estimates and related data sources.
Supports multiple census geometry types.
"""

import pandas as pd 
import numpy as np 

import os 
import sys 
from typing import Union

import geopandas as gpd 
import matplotlib.pyplot as plt 


import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

from shapely import wkt

from logger import setup_logger 
from geometry_config import GeometryType, get_geometry_paths

logger = setup_logger("map-generation-subroutine")
logger.setLevel("INFO")



LATEX=False

SELECT_TOP_N = False
TOP_N_TO_SELECT = 393
LIVE_LOAD_NEXAR_DATA=False

WGS='EPSG:4326'
PROJ='EPSG:2263'


if LATEX: 
    # enable latex plotting 
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    logger.info("LaTeX plotting enabled")


def generate_maps(
    run_id, 
    estimate_path, 
    estimate='at_least_one_positive_image_by_area',
    geometry_type: Union[GeometryType, str] = GeometryType.CT
):
    """
    Generate flood visualization maps.
    
    Parameters
    ----------
    run_id : str
        Run identifier for output directory
    estimate_path : str
        Path to estimate CSV file
    estimate : str
        Estimate column name to visualize
    geometry_type : GeometryType or str
        Census geometry type (ct, cbg, cb)
    """
    # Handle geometry type
    if isinstance(geometry_type, str):
        geometry_type = GeometryType(geometry_type.lower())
    
    paths = get_geometry_paths(geometry_type)
    config = paths.config
    id_column = config.id_column
    
    logger.info(f"Generating maps for {config.display_name}s")

    if LIVE_LOAD_NEXAR_DATA:

        logger.info("Loading raw Nexar data for September 29 dashcam flooding data.")

        entire_sep29 = pd.read_csv("data/processed/entire_sep29_all.csv", engine='pyarrow')
        entire_sep29['frame_id'] = entire_sep29['image_path'].apply(lambda x: x.split('/')[-1].split('.')[0])
        entire_sep29

        sep29_md = pd.read_csv("data/processed/2023-09-29_md.csv", engine='pyarrow')
        sep29_md['frame_id'] = sep29_md['frame_id'].apply(lambda x: "nlbx_"+x)

        entire_sep29 = entire_sep29.merge(sep29_md, on='frame_id', how='left')
        entire_sep29 = gpd.GeoDataFrame(entire_sep29, geometry=wkt.loads(entire_sep29['geometry']), crs=PROJ)

        sep29_positives = entire_sep29[entire_sep29['sentiment_1'] == 1]

        inspection_set_annotated = pd.read_csv("data/processed/inspection_set.csv")
        inspection_set_annotated['frame_id'] = inspection_set_annotated['image'].apply(lambda x: 'nlbx_'+x.split('/')[-1].split('.')[0].split('_')[-1])
        inspection_set_annotated['choice'] = inspection_set_annotated['choice'].apply(lambda x: 1 if x == 'Flooded road' else 0)
        # drop everything except frame_id and choice 
        inspection_set_annotated['pred'] = inspection_set_annotated['sentiment_1']
        inspection_set_annotated['tp'] = ((inspection_set_annotated['choice'] == 1) & (inspection_set_annotated['pred'] == 1)).astype(int)
        inspection_set_annotated['fp'] = ((inspection_set_annotated['choice'] == 0) & (inspection_set_annotated['pred'] == 1)).astype(int)
        inspection_set_annotated['tn'] = ((inspection_set_annotated['choice'] == 0) & (inspection_set_annotated['pred'] == 0)).astype(int)
        inspection_set_annotated['fn'] = ((inspection_set_annotated['choice'] == 1) & (inspection_set_annotated['pred'] == 0)).astype(int)
        inspection_set_annotated = inspection_set_annotated[['frame_id', 'choice', 'pred', 'tp', 'fp', 'tn', 'fn']]

        logger.info(f"Loaded and processed inspection set annotations with {len(inspection_set_annotated)} annotations.")

        len_before = len(entire_sep29)
        entire_sep29 = entire_sep29.merge(inspection_set_annotated, on='frame_id', how='left')
        assert len(entire_sep29) == len_before

        sep29_gt = entire_sep29[entire_sep29['choice'] == 1]

    
    else: 
        logger.info("Loading local preprocessed data for September 29 dashcam flooding data.")

        sep29_positives = pd.read_csv("data/processed/sep29_positives.csv", engine='pyarrow')
        sep29_positives = gpd.GeoDataFrame(sep29_positives, geometry=sep29_positives.geometry.apply(lambda x: wkt.loads(x)), crs=PROJ)

        sep29_gt = pd.read_csv("data/processed/sep29_gt.csv", engine='pyarrow')
        sep29_gt = gpd.GeoDataFrame(sep29_gt, geometry=sep29_gt.geometry.apply(lambda x: wkt.loads(x)), crs=PROJ)


    logger.info("Loaded and processed september 29 dashcam flooding data.")

    # Load geometry-appropriate analysis set
    flooding_dataset_path = paths.flooding_dataset_path
    if flooding_dataset_path.exists():
        analysis_set = pd.read_csv(flooding_dataset_path)
    else:
        # Fallback to CT dataset for backward compatibility
        analysis_set = pd.read_csv("data/processed/flooding_ct_dataset.csv")
        logger.warning(f"Using fallback CT dataset, {flooding_dataset_path} not found")

    logger.success("Loaded analysis set.")



    estimate_df = pd.read_csv(estimate_path, engine='pyarrow')
    # Handle both geoid and tract_id columns
    id_col = 'geoid' if 'geoid' in estimate_df.columns else 'tract_id'
    estimate_df[id_col] = estimate_df[id_col].astype(int)

    logger.info("Loaded estimates from ICAR model.")


    analysis_set = analysis_set.merge(estimate_df, left_on=id_column, right_on=id_col, how='left').drop_duplicates(subset=id_column)
    analysis_set = gpd.GeoDataFrame(analysis_set, geometry=analysis_set.geometry.apply(lambda x: wkt.loads(x)), crs=PROJ)

    logger.success("Merged model estimates with analysis set.")


    # Load geometry-appropriate geojson
    geojson_path = paths.aggregation_geojson_path
    if not geojson_path.exists():
        geojson_path = paths.geojson_path
    
    geo_df = gpd.read_file(str(geojson_path))
    geo_df = geo_df.to_crs(PROJ)

    logger.info(f"Loaded NYC {config.display_name} data ({len(geo_df)} areas).")

    nyc_311 = pd.read_csv('aggregation/flooding/data/nyc311_flooding_sep29.csv').dropna(subset=['latitude', 'longitude'])
    nyc_311 = gpd.GeoDataFrame(nyc_311, geometry=gpd.points_from_xy(nyc_311.longitude, nyc_311.latitude), crs=WGS).to_crs(PROJ)
    
    logger.info("Loaded and filtered 311 complaints for September 29, 2023.")


    # FLOODNET 

    all_floodnet_sensor_geo = pd.read_csv('aggregation/flooding/static/current_floodnet_sensors.csv')
    all_floodnet_sensor_geo = gpd.GeoDataFrame(all_floodnet_sensor_geo, geometry=gpd.points_from_xy(all_floodnet_sensor_geo.longitude, all_floodnet_sensor_geo.latitude), crs=WGS).to_crs(PROJ)


    logger.info("Loaded and processed Floodnet sensor data.")


    # DEP STORMWATER 
    moderate_current_conditions = gpd.read_file('aggregation/flooding/static/dep_stormwater_moderate_current/data.gdb').to_crs(PROJ)
    moderate_current_conditions.describe()

    logger.info("Loaded and processed DEP stormwater moderate current conditions data.")


    geo_enriched = geo_df.copy() 

    # get nearest 311 complaint to each area 
    geo_enriched = gpd.sjoin_nearest(geo_enriched, nyc_311, distance_col='nearest_report_to_area')
    # drop index_left, index_right, dont fail if they dont exist
    geo_enriched.drop(columns=['index_right'], errors='ignore', inplace=True)


    # get nearest floodnet sensor to each area
    geo_enriched = gpd.sjoin_nearest(geo_enriched, all_floodnet_sensor_geo, distance_col='nearest_sensor_to_area')
    # drop index_left, index_right, dont fail if they dont exist
    geo_enriched.drop(columns=['index_right'], errors='ignore', inplace=True)

    # get nearest '1' flooding area to each area
    stormwater_filter = moderate_current_conditions['Flooding_Category'] == 1
    geo_enriched = gpd.sjoin_nearest(geo_enriched, moderate_current_conditions[stormwater_filter], distance_col='nearest_nuisance_flooding_area')
    # drop index_left, index_right, dont fail if they dont exist
    geo_enriched.drop(columns=['index_right'], errors='ignore', inplace=True)

    # get nearest '2' flooding area to each area
    stormwater_filter = moderate_current_conditions['Flooding_Category'] == 2
    geo_enriched = gpd.sjoin_nearest(geo_enriched, moderate_current_conditions[stormwater_filter], distance_col='nearest_deep_flooding_area')
    # drop index_left, index_right, dont fail if they dont exist
    geo_enriched.drop(columns=['index_right'], errors='ignore', inplace=True)

    # Find label column for grouping
    label_col = 'CTLabel' if 'CTLabel' in geo_df.columns else id_column
    
    nyc_311 = gpd.sjoin_nearest(nyc_311, geo_df, distance_col='nearest_complaint_to_area')
    # drop index_right 
    nyc_311.drop(columns=['index_right'], inplace=True)

    # drop duplicate rows on id column
    geo_enriched = geo_enriched.drop_duplicates(subset=id_column)

    # count complaints per area
    if label_col in nyc_311.columns:
        geo_enriched['n_complaints'] = geo_enriched[label_col].map(nyc_311.groupby(label_col).size()).fillna(0)
    else:
        geo_enriched['n_complaints'] = geo_enriched[id_column].map(nyc_311.groupby(id_column).size()).fillna(0)


    estimate_by_geo = analysis_set.groupby(id_column)[estimate].mean().reset_index()
    estimate_by_geo[id_column] = estimate_by_geo[id_column].astype(int)
    geo_enriched[id_column] = geo_enriched[id_column].astype(int)
    # merge with geo_enriched
    geo_enriched = geo_enriched.merge(estimate_by_geo, on=id_column, how='left')



    # count frames per area 
    if SELECT_TOP_N:
    # if inferred_p_y is in the top N, then mark classified_positive as 1. else 0 
        geo_enriched['classified_positive'] = geo_enriched[estimate].rank(ascending=False, method='first') <= TOP_N_TO_SELECT
    else:
        geo_enriched['classified_positive'] = geo_enriched[estimate]

    geo_enriched['classified_postiive'] = geo_enriched['classified_positive'].astype(float)
    logger.success(f"Enriched {config.display_name} data with model estimates and other flooding data sources.")


    # Define opacity levels
    NYC311_ALPHA = 1
    SENSOR_ALPHA = 1
    FLOODING_AREA_ALPHA = 1
    VP_ALPHA = 0.7

    PAIRED = True

    # Basemap color
    ocean = '#99b3cc'

    # Boroughs to iterate through
    BOROUGHS = ['']

    for BORO in BOROUGHS:
        if BORO == '':
            geo_enriched_for_plot = geo_enriched
        else:
            boro_col = 'BoroName' if 'BoroName' in geo_enriched.columns else None
            if boro_col:
                geo_enriched_for_plot = geo_enriched[geo_enriched[boro_col] == BORO]
            else:
                geo_enriched_for_plot = geo_enriched

        for i in range(1, 5):  # Iterating through layers to plot
            fig, ax = plt.subplots(figsize=(25, 25))
            
            norm = None
            if estimate == 'p_y': 
                logger.info("Using lognorm for p_y")
                # use lognorm for p_y 
                from matplotlib.colors import LogNorm
                norm = LogNorm(vmin=geo_enriched_for_plot['classified_positive'].quantile(0.003), vmax=geo_enriched_for_plot['classified_positive'].quantile(0.997))
                # plot layer with areas, colored by classified_positive 
                geo_enriched_for_plot.plot(
                    ax=ax, 
                    column='classified_positive', 
                    cmap='coolwarm', 
                    norm=norm,
                    alpha=0.5, 
                    edgecolor='white', 
                    linewidth=0.5, 
                    zorder=2, 
                    legend=True, 
                    legend_kwds={
                        'label': f'P({estimate})', 
                        'orientation': 'horizontal', 
                        'pad': 0.01, 
                        'aspect': 50, 
                        'shrink': 0.5, 
                        'extend': 'neither', 
                        'format': '%.3f'
                    }
                )
            else: 
                # plot layer with areas, colored by classified_positive 
                geo_enriched_for_plot.plot(
                    ax=ax, 
                    column='classified_positive', 
                    cmap='coolwarm', 
                    alpha=0.5, 
                    edgecolor='white', 
                    linewidth=0.5, 
                    zorder=2, 
                    legend=True, 
                    legend_kwds={
                        'label': f'P({estimate})', 
                        'orientation': 'horizontal', 
                        'pad': 0.01, 
                        'aspect': 50, 
                        'shrink': 0.5, 
                        'extend': 'neither', 
                        'format': '%.3f'
                    }
                )
            


            

            sep29_positives.plot(ax=ax, color='orange', marker='o', alpha=0.25, markersize=25, zorder=6, label='Classified Positive Image')
            sep29_gt.plot(ax=ax, color='red', marker='o', alpha=0.25, markersize=25, zorder=6, label='Ground Truth Positive Image')

            # Conditional plotting based on PAIRED setting
            if (i == 2 and PAIRED) or not PAIRED:
                all_floodnet_sensor_geo.plot(ax=ax, color='darkviolet', marker='D', alpha=SENSOR_ALPHA, markersize=56, zorder=5, label='Floodnet Sensor')
            
            if (i == 3 and PAIRED) or not PAIRED:
                nyc_311.plot(ax=ax, color='brown', marker='^', alpha=NYC311_ALPHA, markersize=52, zorder=4, label='311 Complaint')
            
            if (i == 4 and PAIRED) or not PAIRED:
                moderate_current_conditions[moderate_current_conditions['Flooding_Category'] == 1].plot(ax=ax, color=ocean, edgecolor=ocean, alpha=FLOODING_AREA_ALPHA, linewidth=2, zorder=3)
                moderate_current_conditions[moderate_current_conditions['Flooding_Category'] == 2].plot(ax=ax, edgecolor='darkblue', color='darkblue', alpha=FLOODING_AREA_ALPHA, linewidth=2, zorder=3)
                
                # Create new legend handles for custom stormwater legend
                blue_line = mlines.Line2D([], [], color=ocean, markersize=15, label='Nuisance Flooding Area')
                darkblue_line = mlines.Line2D([], [], color='darkblue', markersize=15, label='Deep Flooding Area')

                # Add the new handles to the existing ones
                existing_handles, existing_labels = ax.get_legend_handles_labels()
                updated_handles = existing_handles + [blue_line, darkblue_line]
                updated_labels = existing_labels + ['Nuisance Flooding Area', 'Deep Flooding Area']

                # Create the new legend
                legend = ax.legend(handles=updated_handles, labels=updated_labels, loc='upper left', fontsize=28, scatterpoints=1, fancybox=True, framealpha=1).set_zorder(7)
            else:
                legend = ax.legend(loc='upper left', fontsize=28, framealpha=1, scatterpoints=1).set_zorder(7)
            
            # Setting bounds
            bounds = geo_enriched_for_plot.total_bounds
            ax.set_xlim([bounds[0], bounds[2]])
            ax.set_ylim([bounds[1], bounds[3]])

            # Turn off the axes
            ax.axis('off')

            # Saving figures
            os.makedirs(f'runs/{run_id}/maps', exist_ok=True)
            geo_prefix = geometry_type.value
            path = f'runs/{run_id}/maps/nyc_flooding_map_{geo_prefix}_{estimate}_{PAIRED}_{BORO}_{i}_noagg_zoomin.pdf' if BORO != '' else f'runs/{run_id}/maps/nyc_flooding_map_{geo_prefix}_{estimate}_{PAIRED}_{i}_noagg.pdf'
            plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.0)
            plt.close()

            logger.info(f"Generated map for {BORO} - {i}")

    logger.success("Completed map generation.")

if __name__ == '__main__':
    run_id = sys.argv[1]
    estimate_path = sys.argv[2]
    estimate = sys.argv[3]
    geometry_type = sys.argv[4] if len(sys.argv) > 4 else 'ct'

    generate_maps(run_id, estimate_path, estimate=estimate, geometry_type=geometry_type)


