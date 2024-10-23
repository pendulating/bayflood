import os 
import rasterio


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

        # Write the new raster to disk
        if REGEN_TOPOLOGY: 
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            with rasterio.open(OUTPUT_PATH, "w", **new_meta) as dst:
                dst.write(topology)

    