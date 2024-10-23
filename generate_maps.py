
import pandas as pd 
import numpy as np 

import os 
import sys 

import geopandas as gpd 
import matplotlib.pyplot as plt 


import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

from shapely import wkt

from logger import setup_logger 

logger = setup_logger("map-generation-subroutine")
logger.setLevel("INFO")



LATEX=True

SELECT_TOP_N = False
TOP_N_TO_SELECT = 393
LIVE_LOAD_NEXAR_DATA=False


if LATEX: 
    # enable latex plotting 
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    logger.info("LaTeX plotting enabled")


def generate_maps(run_id, estimate_path, estimate='at_least_one_positive_image_by_area'):

    if LIVE_LOAD_NEXAR_DATA:

        entire_sep29 = pd.read_csv("notebooks/cambrian/entire_sep29_all.csv", engine='pyarrow')
        entire_sep29['frame_id'] = entire_sep29['image_path'].apply(lambda x: x.split('/')[-1].split('.')[0])
        entire_sep29

        sep29_md = pd.read_csv("/share/ju/urban-fingerprinting/output/default/df/2023-09-29/md.csv", engine='pyarrow')
        sep29_md['frame_id'] = sep29_md['frame_id'].apply(lambda x: "nlbx_"+x)

        entire_sep29 = entire_sep29.merge(sep29_md, on='frame_id', how='left')
        entire_sep29 = gpd.GeoDataFrame(entire_sep29, geometry=wkt.loads(entire_sep29['geometry']), crs='EPSG:2263')

        sep29_positives = entire_sep29[entire_sep29['sentiment_1'] == 1]
    
    else: 

        sep29_positives = pd.read_csv("data/processed/sep29_positives.csv", engine='pyarrow')
        sep29_positives = gpd.GeoDataFrame(sep29_positives, geometry=sep29_positives.geometry.apply(lambda x: wkt.loads(x)), crs='EPSG:2263')


    logger.info("Loaded and processed september 29 dashcam flooding data.")


    analysis_set = pd.read_csv("data/processed/flooding_ct_dataset.csv")


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

    logger.success("Completed merge of inspection set annotations with september 29 dashcam flooding data.")



    estimate_df = pd.read_csv(estimate_path, engine='pyarrow')
    estimate_df['tract_id'] = estimate_df['tract_id'].astype(int)

    logger.info("Loaded estimates from ICAR model.")


    analysis_set = analysis_set.merge(estimate_df, left_on='GEOID', right_on='tract_id', how='left').drop_duplicates(subset='GEOID')
    analysis_set = gpd.GeoDataFrame(analysis_set, geometry=analysis_set.geometry.apply(lambda x: wkt.loads(x)), crs='EPSG:2263')

    logger.success("Merged model estimates with analysis set.")



    nyc_ct = gpd.read_file('aggregation/geo/data/ct-nyc-2020.geojson')
    nyc_ct = nyc_ct.to_crs(2263)

    logger.info("Loaded NYC census tract data.")

    nyc_311 = pd.read_csv('aggregation/flooding/data/nyc311_flooding_sep29.csv').dropna(subset=['latitude', 'longitude'])
    nyc_311 = gpd.GeoDataFrame(nyc_311, geometry=gpd.points_from_xy(nyc_311.longitude, nyc_311.latitude), crs='EPSG:4326').to_crs(2263)
    
    logger.info("Loaded and filtered 311 complaints for September 29, 2023.")


    # FLOODNET 
    all_floodnet_sensor_geo = gpd.GeoDataFrame(all_floodnet_sensors_geo, geometry=gpd.points_from_xy(all_floodnet_sensors_geo.lon, all_floodnet_sensors_geo.lat), crs='EPSG:4326').to_crs(2263)

    all_floodnet_sensor_geo = pd.read_csv('aggregation/flooding/static/floodnet_sensor_coordinates.csv')
    all_floodnet_sensor_geo = gpd.GeoDataFrame(all_floodnet_sensors_geo, geometry=gpd.points_from_xy(all_floodnet_sensors_geo.lon, all_floodnet_sensors_geo.lat), crs='EPSG:4326').to_crs(2263)


    logger.info("Loaded and processed Floodnet sensor data.")


    # DEP STORMWATER 
    moderate_current_conditions = gpd.read_file('aggregation/flooding/data/NYCFloodStormwaterFloodMaps/NYC Stormwater Flood Map - Moderate Flood (2.13 inches per hr) with Current Sea Levels/NYC_Stormwater_Flood_Map_Moderate_Flood_2_13_inches_per_hr_with_Current_Sea_Levels.gdb').to_crs(PROJ)
    moderate_current_conditions.describe()

    logger.info("Loaded and processed DEP stormwater moderate current conditions data.")


    ct_enriched = nyc_ct.copy() 

    # get nearest 311 complaint to each tract 
    ct_enriched = gpd.sjoin_nearest(ct_enriched, nyc_311, distance_col='nearest_report_to_ct')
    # drop index_left, index_right, dont fail if they dont exist
    ct_enriched.drop(columns=['index_right'], errors='ignore', inplace=True)


    # get nearest floodnet sensor to each tract
    ct_enriched = gpd.sjoin_nearest(ct_enriched, all_floodnet_sensor_geo, distance_col='nearest_sensor_to_ct')
    # drop index_left, index_right, dont fail if they dont exist
    ct_enriched.drop(columns=['index_right'], errors='ignore', inplace=True)

    # get nearest '1' flooding area to each tract
    stormwater_filter = moderate_current_conditions['Flooding_Category'] == 1
    ct_enriched = gpd.sjoin_nearest(ct_enriched, moderate_current_conditions[stormwater_filter], distance_col='nearest_nuisance_flooding_area_to_ct')
    # drop index_left, index_right, dont fail if they dont exist
    ct_enriched.drop(columns=['index_right'], errors='ignore', inplace=True)

    # get nearest '2' flooding area to each tract
    stormwater_filter = moderate_current_conditions['Flooding_Category'] == 2
    ct_enriched = gpd.sjoin_nearest(ct_enriched, moderate_current_conditions[stormwater_filter], distance_col='nearest_deep_flooding_area_to_ct')
    # drop index_left, index_right, dont fail if they dont exist
    ct_enriched.drop(columns=['index_right'], errors='ignore', inplace=True)

    nyc_311 = gpd.sjoin_nearest(nyc_311, nyc_ct, distance_col='nearest_complaint_to_ct')
    # drop index_right 
    nyc_311.drop(columns=['index_right'], inplace=True)

    # drop duplicate rows on CTLabel
    ct_enriched = ct_enriched.drop_duplicates(subset='GEOID')

    # count complaints per ct
    ct_enriched['n_complaints'] = ct_enriched['CTLabel'].map(nyc_311.groupby('CTLabel').size()).fillna(0)


    estimate_by_ct = analysis_set.groupby('GEOID')[estimate].mean().reset_index()
    estimate_by_ct['GEOID'] = estimate_by_ct['GEOID'].astype(int)
    ct_enriched['GEOID'] = ct_enriched['GEOID'].astype(int)
    # merge with ct_enriched
    ct_enriched = ct_enriched.merge(estimate_by_ct, left_on='GEOID', right_on='GEOID', how='left')



    # count frames per ct 
    if SELECT_TOP_N:
    # if inferred_p_y is in the top N, then mark classified_positive as 1. else 0 
        ct_enriched['classified_positive'] = ct_enriched[estimate].rank(ascending=False, method='first') <= TOP_N_TO_SELECT
    else:
        ct_enriched['classified_positive'] = ct_enriched[estimate]

    ct_enriched['classified_postiive'] = ct_enriched['classified_positive'].astype(float)
    logger.success("Enriched census tract data with model estimates and other flooding data sources.")


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
            ct_enriched_for_plot = ct_enriched
        else:
            ct_enriched_for_plot = ct_enriched[ct_enriched['BoroName'] == BORO]

        for i in range(1, 5):  # Iterating through layers to plot
            fig, ax = plt.subplots(figsize=(25, 25))
            
            norm = None
            if estimate == 'p_y': 
                logger.info("Using lognorm for p_y")
                # use lognorm for p_y 
                from matplotlib.colors import LogNorm
                norm = LogNorm()
                # plot layer with census tracts, colored by classified_positive 
                ct_enriched_for_plot.plot(
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
                # plot layer with census tracts, colored by classified_positive 
                ct_enriched_for_plot.plot(
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
            bounds = ct_enriched_for_plot.total_bounds
            ax.set_xlim([bounds[0], bounds[2]])
            ax.set_ylim([bounds[1], bounds[3]])

            # Turn off the axes
            ax.axis('off')

            # Saving figures
            os.makedirs(f'runs/{run_id}/maps', exist_ok=True)
            path = f'runs/{run_id}/maps/nyc_flooding_map_paired_{estimate}_{PAIRED}_{BORO}_{i}_noagg_zoomin.png' if BORO != '' else f'runs/{run_id}/maps/nyc_flooding_map_paired_{estimate}_{PAIRED}_{i}_noagg.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.0)
            plt.close()

            logger.info(f"Generated map for {BORO} - {i}")

    logger.success("Completed map generation.")





