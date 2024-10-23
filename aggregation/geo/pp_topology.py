import os 
import rasterio
from rasterio.enums import Resampling
from rasterstats import zonal_stats
from itertools import islice
import pandas as pd
import geopandas as gpd 

import sys 
sys.path.append('../../')
from logger import setup_logger
logger = setup_logger('pp-topology-subroutine')
logger.setLevel("INFO")


def downsample_raster(TOPOGRAPHY_NYC, downsample_factor=10, REGEN_TOPOLOGY=False, OUTPUT_PATH=''):

    # Open the raster
    with rasterio.open(TOPOGRAPHY_NYC) as src:
        # Calculate new transform and dimensions
        new_transform = src.transform * src.transform.scale(
            downsample_factor,
            downsample_factor
        )
        new_width = src.width // downsample_factor
        new_height = src.height // downsample_factor
        
        # Resample the raster
        topology = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.bilinear
        )

        # Create a new rasterio-like object with updated metadata
        new_meta = src.meta.copy()
        new_meta.update({
            "driver": "GTiff",
            "height": new_height,
            "width": new_width,
            "transform": new_transform
        })

        logger.success("Downsampling complete")

        # Write the new raster to disk
        if REGEN_TOPOLOGY: 
            logger.info(f"Writing downsampled topology to {OUTPUT_PATH}")
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            with rasterio.open(OUTPUT_PATH, "w", **new_meta) as dst:
                dst.write(topology)



def sample_topology(topology_path, sampling_geom,save=True):

    # make sure topology_path ends in .tif
    assert topology_path.endswith('.tif'), "topology_path must end in .tif"

    sampling_geom = gpd.read_file(sampling_geom).to_crs("EPSG:2263")

    summary_stats = zonal_stats( 
        sampling_geom,
        topology_path,
    )

    

    # summary_stats is a list of dicts 
    # each dict has the same keys
    # convert summary_stats to df 
    summary_stats_df = pd.DataFrame(summary_stats)
    # join with sampling_geom 
    sampling_geom = sampling_geom[['GEOID']]
    summary_stats_df = sampling_geom.join(summary_stats_df)

    # drop 'count' 
    summary_stats_df = summary_stats_df.drop(columns='count')

    logger.success("Topology sampling complete")

    if save: 
        logger.info("Saving topology sampling to data/processed/ct_nyc_topology.csv")
        summary_stats_df.to_csv('data/processed/ct_nyc_topology.csv')




if __name__ == '__main__': 

    downsample_raster("data/DEM_LiDAR_1ft_2010_Improved_NYC_int.tif", downsample_factor=10, REGEN_TOPOLOGY=True, OUTPUT_PATH='data/processed/topology_nyc_downsampled.tif')
    sample_topology('data/processed/topology_nyc_downsampled.tif', 'data/ct-nyc-wi-2020.geojson', save=True)

    logger.success("Topology sampling complete")