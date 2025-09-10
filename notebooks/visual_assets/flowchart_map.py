# %%
import geopandas as gpd 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import logging
import sys
from datetime import datetime
import os
import matplotlib as mpl
from geodatasets import get_path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import box

# Configure logging
def setup_logger():
    logger = logging.getLogger('flowchart_map')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(f'flowchart_map_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

# %%
logger.info("Loading census tract data...")
ct_nyc = gpd.read_file('../../aggregation/geo/data/ct-nyc-2020.geojson')
logger.info(f"Loaded {len(ct_nyc)} census tracts")

logger.info("Loading nybb...")
nybb = gpd.read_file(get_path("nybb")).to_crs("EPSG:4326")
logger.info(f"Loaded the new york city boroughs from geopandas.")

# %%
logger.info("Loading inspection set data...")
inspection_set = pd.read_csv("/share/ju/matt/street-flooding/notebooks/cambrian/inspection_set_annotated.csv")
inspection_set['frame_id'] = inspection_set['image'].str.replace('/data/local-files/?d=', '')
inspection_set['frame_id'] = inspection_set['frame_id'].apply(lambda x: str(x.split('/')[-1].split('_')[1].split('.')[0]))
logger.info(f"Loaded {len(inspection_set)} inspection records")

# %%
logger.info("Loading September 29 classification data...")
sep29_cls = pd.read_csv("/share/ju/matt/street-flooding/notebooks/cambrian/entire_sep29_all.csv", engine='pyarrow')
sep29_cls['frame_id'] = sep29_cls['image_path'].apply(lambda x: str(x.split('/')[-1].split('_')[1].split('.')[0]))
logger.info(f"Loaded {len(sep29_cls)} classification records")
logger.info(f"Value counts of sentiment_1: {sep29_cls['sentiment_1'].value_counts()}")

# %%
logger.info("Loading September 29 DSI data...")
sep29_dsi = pd.read_csv("/share/ju/matt/street-flooding/data/md.csv", engine='pyarrow')
sep29_dsi = gpd.GeoDataFrame(sep29_dsi, geometry=gpd.points_from_xy(sep29_dsi['gps_info.longitude'], sep29_dsi['gps_info.latitude']), crs='EPSG:4326')
logger.info(f"Loaded {len(sep29_dsi)} DSI records")

# %%
logger.info("Merging September 29 classification data with DSI data...")
sep29_cls = sep29_dsi.merge(sep29_cls, on='frame_id', how='left')
logger.info(f"Merged dataset contains {len(sep29_cls)} records")

sep29_positives = sep29_cls[sep29_cls['sentiment_1'] == 1]

# %%
logger.info("Merging inspection set with DSI data...")
inspection_set = inspection_set.merge(sep29_dsi, on='frame_id', how='left')
inspection_set = gpd.GeoDataFrame(inspection_set, geometry=inspection_set.geometry, crs='EPSG:4326')
inspection_set['choice'] = inspection_set['choice'].str.contains('Flooded')
logger.info(f"Merged inspection set contains {len(inspection_set)} records")
logger.info(f"Inspection set GT value counts: {inspection_set['choice'].value_counts()}")

# %% 
logger.info("Loading BayFlood flood risk data...")
bayflood = pd.read_csv("/share/ju/matt/street-flooding/runs/icar_icar/simulated_False/ahl_True/covariates_True/FEB7_FINAL_KDD_MODEL_20250207-1732/analysis_df_FEB7_FINAL_KDD_MODEL_02072025.csv")
ct_nyc['p_y'] = bayflood['p_y']
logger.info(f"Loaded {len(ct_nyc)} census tracts with BayFlood flood risk data")

# %%
logger.info("Loading FloodNet sensor data...")
floodnet_current = pd.read_csv("/share/ju/matt/street-flooding/aggregation/flooding/static/current_floodnet_sensors.csv")
floodnet_current = gpd.GeoDataFrame(floodnet_current, geometry=gpd.points_from_xy(floodnet_current.longitude, floodnet_current.latitude), crs="EPSG:4326")
logger.info(f"Loaded {len(floodnet_current)} FloodNet sensors")

# %%
logger.info("Loading September 29 311 data...")
sep29_311 = pd.read_csv("/share/ju/matt/street-flooding/aggregation/flooding/data/nyc311_flooding_sep29.csv")
sep29_311 = gpd.GeoDataFrame(sep29_311, geometry=gpd.points_from_xy(sep29_311.longitude, sep29_311.latitude), crs="EPSG:4326")
# drop na geometry
sep29_311 = sep29_311.dropna(subset=['geometry', 'latitude', 'longitude'])
logger.info(f"Loaded {len(sep29_311)} 311 records after dropping NA geometries")

# %%
logger.info("Loading moderate depth flood map data...")
moderate_dep = gpd.read_file("/share/ju/matt/street-flooding/aggregation/flooding/data/NYCFloodStormwaterFloodMaps/NYC Stormwater Flood Map - Moderate Flood (2.13 inches per hr) with Current Sea Levels/NYC_Stormwater_Flood_Map_Moderate_Flood_2_13_inches_per_hr_with_Current_Sea_Levels.gdb")
moderate_dep = moderate_dep.to_crs("EPSG:4326")

# Explode MultiPolygons into individual Polygons and buffer them
logger.info("Exploding and buffering moderate depth flood map geometries...")
exploded_geoms = []
for idx, row in moderate_dep.iterrows():
    if row.geometry.geom_type == 'MultiPolygon':
        for poly in row.geometry.geoms:
            exploded_geoms.append(poly.buffer(0.0001))  # Buffer by ~10 meters
    else:
        exploded_geoms.append(row.geometry.buffer(0.0001))

# Create new GeoDataFrame with exploded and buffered geometries
moderate_dep = gpd.GeoDataFrame(geometry=exploded_geoms, crs=moderate_dep.crs)
logger.info(f"Processed {len(moderate_dep)} flood map geometries")

def save_layer(gdf, layer_name, output_dir='layer_outputs', style=None):
    """
    Save a single layer visualization with consistent settings.
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        The layer data to visualize
    layer_name : str
        Name of the layer to save
    output_dir : str
        Directory to save the output files
    style : dict
        Style parameters for the layer. Can include:
        - polygon_color: color for polygons
        - polygon_alpha: alpha for polygons
        - polygon_edge_color: color for polygon edges
        - polygon_edge_width: width of polygon edges
        - point_color: color for points
        - point_size: size of points
        - point_alpha: alpha for points
        - line_color: color for lines
        - line_width: width of lines
        - line_alpha: alpha for lines
        - color_by: column to color by
        - colormap: colormap to use
        - use_lognorm: whether to use lognorm for color scaling (default: False)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure the layer is in the same CRS as nybb
    if gdf.crs != nybb.crs:
        print("layer not in nybb crs, converting...")
        gdf = gdf.to_crs(nybb.crs)
    
    # Get the bounds from nybb (our reference layer)
    bounds = nybb.total_bounds
    
    # Create figure with consistent size and DPI
    fig = plt.figure(figsize=(20, 20), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    
    # Set consistent view parameters
    ax.view_init(elev=10, azim=244)
    ax.set_proj_type('persp')
    
    # Set consistent bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_zlim(0, 1)  # Minimal z-range for 3D effect
    
    # Remove all padding and margins
    ax.margins(0)
    ax.set_axis_off()
    
    # Default style parameters
    default_style = {
        'polygon_color': 'lightblue',
        'polygon_alpha': 0.7,
        'polygon_edge_color': 'none',
        'polygon_edge_width': 0.0,
        'point_color': 'red',
        'point_size': 20,
        'point_alpha': 0.9,
        'line_color': 'blue',
        'line_width': 2,
        'line_alpha': 0.8,
        'use_lognorm': False,  # New parameter for lognorm support
    }
    
    # Merge default style with provided style
    style = {**default_style, **(style or {})}
    
    # Set up colormap if color_by is specified
    color_by = style.get('color_by', None)
    colormap = style.get('colormap', 'Reds')
    norm = None
    cmap = None
    if color_by and color_by in gdf.columns:
        values = gdf[color_by].values
        cmap = mpl.cm.get_cmap(colormap)
        if style['use_lognorm']:
            # Handle zero and negative values for lognorm
            min_val = np.nanmin(values[values > 0]) if np.any(values > 0) else 1e-10
            norm = mpl.colors.LogNorm(vmin=min_val, vmax=np.nanmax(values))
        else:
            norm = mpl.colors.Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
    
    # Collect points for batch processing
    points_x = []
    points_y = []
    points_colors = []
    
    # Process geometries
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type in ['Polygon', 'MultiPolygon']:
            if geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    x, y = poly.exterior.xy
                    z = np.zeros(len(x))
                    verts = [list(zip(x, y, z))]
                    # Get color based on color_by if specified
                    if color_by and color_by in row:
                        color = cmap(norm(row[color_by])) if cmap is not None else style['polygon_color']
                    else:
                        color = style['polygon_color']
                    poly3d = Poly3DCollection(verts,
                                            alpha=style['polygon_alpha'],
                                            facecolor=color,
                                            edgecolor=style['polygon_edge_color'],
                                            linewidth=style['polygon_edge_width'])
                    ax.add_collection3d(poly3d)
            else:
                x, y = geom.exterior.xy
                z = np.zeros(len(x))
                verts = [list(zip(x, y, z))]
                # Get color based on color_by if specified
                if color_by and color_by in row:
                    color = cmap(norm(row[color_by])) if cmap is not None else style['polygon_color']
                else:
                    color = style['polygon_color']
                poly3d = Poly3DCollection(verts,
                                        alpha=style['polygon_alpha'],
                                        facecolor=color,
                                        edgecolor=style['polygon_edge_color'],
                                        linewidth=style['polygon_edge_width'])
                ax.add_collection3d(poly3d)
        elif geom.geom_type in ['Point', 'MultiPoint']:
            if geom.geom_type == 'MultiPoint':
                for point in geom.geoms:
                    points_x.append(point.x)
                    points_y.append(point.y)
                    # Get color based on color_by if specified
                    if color_by and color_by in row:
                        color = cmap(norm(row[color_by])) if cmap is not None else style['point_color']
                    else:
                        color = style['point_color']
                    points_colors.append(color)
            else:
                points_x.append(geom.x)
                points_y.append(geom.y)
                # Get color based on color_by if specified
                if color_by and color_by in row:
                    color = cmap(norm(row[color_by])) if cmap is not None else style['point_color']
                else:
                    color = style['point_color']
                points_colors.append(color)
    
    # Batch plot points
    if points_x:
        ax.scatter(points_x, points_y, np.zeros(len(points_x)),
                  c=points_colors if points_colors else style['point_color'],
                  s=style['point_size'],
                  alpha=style['point_alpha'])
    
    # Save the layer
    output_path = os.path.join(output_dir, f'{layer_name}.png')
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    
    logger.info(f"Saved layer {layer_name} to {output_path}")
    return output_path

# %%
logger.info("Sampling September 29 classification data for performance...")
sep29_cls = sep29_cls.sample(frac=0.2, random_state=42)  # Sample for performance
logger.info(f"Sampled dataset contains {len(sep29_cls)} records")

# %%
logger.info("Preparing layers for visualization...")
layers = {
    'nybb': {
        'gdf': nybb,
        'style': {
            'polygon_color': 'gainsboro',
            'polygon_alpha': 0.8,
        }
    },
    'census_tracts': {
        'gdf': ct_nyc,
        'style': {
            'polygon_color': '#d3d3d3',
            'polygon_alpha': 0.9,
            'polygon_edge_color': '#808080',
            'polygon_edge_width': 0.5,
        }
    },
    'sep29_dsi': {
        'gdf': sep29_cls, 
        'style': {
            'point_color': mpl.cm.get_cmap('cividis')(150),
            'point_size': 1.5,
            'point_alpha': 0.1,
        },
    },
    'sep29_cls': {
        'gdf': sep29_cls,
        'style': {
            'point_color': mpl.cm.get_cmap('cividis')(150),
            'point_size': 1.5,
            'point_alpha': 0.1,
        }
    },
    'sep29_positives': {
        'gdf': sep29_positives,
        'style': {
            'point_color': 'violet', 
            'point_size': 5,
            'point_alpha': 0.5,
        }
    },
    'inspection_set': {
        'gdf': inspection_set,
        'style': {
            'color_by': 'choice',
            'colormap': mpl.cm.get_cmap('PRGn').reversed(),
            'point_size': 8,
            'point_alpha': 0.5,
        }
    },
    'inspection_set_positives': {
        'gdf': inspection_set[inspection_set['choice'] == 1],
        'style': {
            'point_color': mpl.cm.get_cmap('PRGn')(0.0),  # Get the purple color from PRGn colormap
            'point_size': 8,
            'point_alpha': 0.5,
        },
    },
    'bayflood': {
        'gdf': ct_nyc,
        'style': {
            'color_by': 'p_y',
            'colormap': 'Reds',
            'polygon_alpha': 0.8,
            'use_lognorm': True,  # Enable lognorm for flood risk visualization
        }
    },
    'moderate_depth_flood_map': {
        'gdf': moderate_dep,
        'style': {
            'polygon_color': '#41b6c4',
            'polygon_alpha': 0.8,
        }
    },
    'floodnet_sensors': {
        'gdf': floodnet_current,
        'style': {
            'point_color': '#0c2c84',
            'point_size': 15,
            'point_alpha': 0.7,
        }
    },
    'sep29_311': {
        'gdf': sep29_311,
        'style': {
            'point_color': '#1a80aa',
            'point_size': 10,
            'point_alpha': 0.25,
        }
    },
}

# Save each layer separately
logger.info("Generating individual layer visualizations...")
output_dir = 'layer_outputs'
for layer_name, layer_info in layers.items():
    logger.info(f"Processing layer: {layer_name}")
    save_layer(layer_info['gdf'], layer_name, output_dir, layer_info['style'])

logger.info("All layers have been saved separately")



# %%



