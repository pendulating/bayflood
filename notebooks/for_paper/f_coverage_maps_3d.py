# %%
import pandas as pd
import numpy as np
import geopandas as gpd
import geodatasets
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib.cm import get_cmap
import colorcet as cc
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import logging
import sys
from datetime import datetime

# enable latex plotting 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import constants as c 
import helpers as h
from logger import setup_logger 
log = setup_logger("generate-3d-flood-risk-coverage-maps")
log.setLevel("INFO")
log.info("Modules loaded.")

# %%
ct_nyc = gpd.read_file(f"{c.GEO_PATH}/ct-nyc-2020.geojson", crs=c.WGS).to_crs(c.PROJ)
log.info("Loaded NYC Census Tracts.")

nybb = gpd.read_file(geodatasets.get_path("nybb"), crs=c.WGS).to_crs(c.PROJ)
log.info("Loaded NYC Boroughs.")

# %%
analysis_df = pd.read_csv(c.CURRENT_DF)
log.info("Analysis dataframe loaded.")

analysis_df = h.add_estimate_cols(analysis_df)
EST_TO_USE = c.ESTIMATE_TO_USE
log.info(f"Using estimate: {EST_TO_USE}")

# %%
analysis_df['any_sensors'] = analysis_df['n_floodnet_sensors'] > 0
log.info(f"Found {analysis_df['any_sensors'].sum()} tracts with at least one FloodNet sensor.")

# get all columns with 311 in name, and sum them up
analysis_df['n_311_requests'] = analysis_df.filter(like='311').sum(axis=1)
log.info(f"Found {analysis_df['n_311_requests'].sum()} 311 requests.")

analysis_df['any_311_report'] = analysis_df['n_311_requests'] > 0
log.info(f"Found {analysis_df['any_311_report'].sum()} tracts with at least one 311 report.")

analysis_df['no_dep_flooding'] = analysis_df['dep_moderate_2_frac'] == 0
log.info(f"Found {analysis_df['no_dep_flooding'].sum()} tracts with no DEP flooding.")

# %%
# merge geometry from ct_nyc into analysis_df
analysis_df['GEOID'] = analysis_df['GEOID'].astype(str)
analysis_df = ct_nyc.merge(analysis_df, on='GEOID')
# make sure analysis_df has the same number of rows as ct_nyc 
if len(analysis_df) != len(ct_nyc):
    log.error(f"Length of analysis_df ({len(analysis_df)}) does not match length of ct_nyc ({len(ct_nyc)}).")
    exit(1)
else: 
    log.info(f"Length of analysis_df ({len(analysis_df)}) matches length of ct_nyc ({len(ct_nyc)}).")

analysis_df = gpd.GeoDataFrame(analysis_df, crs=c.PROJ)

# %%
def save_3d_layer(gdf, layer_name, output_dir='3d_coverage_outputs', style=None):
    """
    Save a single layer visualization with consistent 3D settings.
    
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
        log.info("layer not in nybb crs, converting...")
        gdf = gdf.to_crs(nybb.crs)
    
    # Get the bounds from nybb (our reference layer)
    bounds = nybb.total_bounds
    
    # Create figure with consistent size and DPI
    fig = plt.figure(figsize=(20, 20), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    
    # Set consistent view parameters
    ax.view_init(elev=20, azim=270)
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
        'use_lognorm': False,
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
    
    log.info(f"Saved layer {layer_name} to {output_path}")
    return output_path

# %%
# Create the different coverage map layers
log.info("Creating 3D coverage map layers...")

# Base layer - all census tracts in light grey
base_layer = analysis_df.copy()
base_layer['base_color'] = 'lightgrey'

# Isolated signal layer - tracts with model signal but no other external signals
isolated_tracts = analysis_df[
    (analysis_df[EST_TO_USE]) & 
    (analysis_df['any_sensors'] == 0) & 
    (analysis_df['any_311_report'] == 0) & 
    (analysis_df['no_dep_flooding'] == 1)
].copy()
isolated_tracts['highlight_color'] = 'gold'

# No 311 layer - tracts with model signal but no 311 reports
no_311_tracts = analysis_df[
    (analysis_df[EST_TO_USE]) & 
    (analysis_df['any_311_report'] == 0)
].copy()
no_311_tracts['highlight_color'] = 'purple'

# No FloodNet layer - tracts with model signal but no FloodNet signal
no_floodnet_tracts = analysis_df[
    (analysis_df[EST_TO_USE]) & 
    (analysis_df['any_sensors'] == 0)
].copy()
no_floodnet_tracts['highlight_color'] = 'darkgreen'

# No DEP layer - tracts with model signal but no DEP stormwater prediction
no_dep_tracts = analysis_df[
    (analysis_df[EST_TO_USE]) & 
    (analysis_df['no_dep_flooding'] == 1)
].copy()
no_dep_tracts['highlight_color'] = 'darkred'

# %%
# Define the layers for 3D visualization
layers = {
    'base_census_tracts': {
        'gdf': base_layer,
        'style': {
            'polygon_color': 'lightgrey',
            'polygon_alpha': 0.9,
            'polygon_edge_color': 'white',
            'polygon_edge_width': 0.5,
        }
    },
    'isolated_signal': {
        'gdf': isolated_tracts,
        'style': {
            'polygon_color': 'gold',
            'polygon_alpha': 0.8,
            'polygon_edge_color': 'black',
            'polygon_edge_width': 1.0,
        }
    },
    'no_311': {
        'gdf': no_311_tracts,
        'style': {
            'polygon_color': 'purple',
            'polygon_alpha': 0.8,
            'polygon_edge_color': 'black',
            'polygon_edge_width': 1.0,
        }
    },
    'no_floodnet': {
        'gdf': no_floodnet_tracts,
        'style': {
            'polygon_color': 'darkgreen',
            'polygon_alpha': 0.8,
            'polygon_edge_color': 'black',
            'polygon_edge_width': 1.0,
        }
    },
    'no_dep': {
        'gdf': no_dep_tracts,
        'style': {
            'polygon_color': 'darkred',
            'polygon_alpha': 0.8,
            'polygon_edge_color': 'black',
            'polygon_edge_width': 1.0,
        }
    },
}

# %%
# Save each layer separately
log.info("Generating 3D coverage map visualizations...")
output_dir = f"{c.PAPER_PATH}/figures/3d_coverage_maps"
for layer_name, layer_info in layers.items():
    log.info(f"Processing layer: {layer_name}")
    save_3d_layer(layer_info['gdf'], layer_name, output_dir, layer_info['style'])

log.info("All 3D coverage map layers have been saved")

# %%
# Create combined visualizations with base layer + highlight layer
log.info("Creating combined 3D visualizations...")

def create_combined_3d_map(base_gdf, highlight_gdf, map_name, output_dir):
    """
    Create a combined 3D map with base layer and highlighted tracts
    """
    # Ensure both layers are in the same CRS as nybb
    if base_gdf.crs != nybb.crs:
        base_gdf = base_gdf.to_crs(nybb.crs)
    if highlight_gdf.crs != nybb.crs:
        highlight_gdf = highlight_gdf.to_crs(nybb.crs)
    
    # Get the bounds from nybb
    bounds = nybb.total_bounds
    
    # Create figure
    fig = plt.figure(figsize=(20, 20), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    
    # Set consistent view parameters
    ax.view_init(elev=20, azim=270)
    ax.set_proj_type('persp')
    
    # Set consistent bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_zlim(0, 1)
    
    # Remove all padding and margins
    ax.margins(0)
    ax.set_axis_off()
    
    # Plot base layer
    for idx, row in base_gdf.iterrows():
        geom = row.geometry
        if geom.geom_type in ['Polygon', 'MultiPolygon']:
            if geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    x, y = poly.exterior.xy
                    z = np.zeros(len(x))
                    verts = [list(zip(x, y, z))]
                    poly3d = Poly3DCollection(verts,
                                            alpha=0.9,
                                            facecolor='lightgrey',
                                            edgecolor='black',
                                            linewidth=0.6)
                    ax.add_collection3d(poly3d)
            else:
                x, y = geom.exterior.xy
                z = np.zeros(len(x))
                verts = [list(zip(x, y, z))]
                poly3d = Poly3DCollection(verts,
                                        alpha=0.9,
                                        facecolor='lightgrey',
                                        edgecolor='black',
                                        linewidth=0.6)
                ax.add_collection3d(poly3d)

       # Plot NYC borough boundary (simplified approach)
    # Ensure nybb is in the same CRS as the plot
    if nybb.crs != base_gdf.crs:
        nybb_plot = nybb.to_crs(base_gdf.crs)
    else:
        nybb_plot = nybb
    
    # Plot borough boundaries as simple lines
    for idx, row in nybb_plot.iterrows():
        geom = row.geometry
        if geom is not None and not geom.is_empty:
            if geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    if poly is not None and not poly.is_empty:
                        x, y = poly.exterior.xy
                        z = np.zeros(len(x))
                        ax.plot(x, y, z, color='black', linewidth=3, alpha=1.0)
            elif geom.geom_type == 'Polygon':
                x, y = geom.exterior.xy
                z = np.zeros(len(x))
                ax.plot(x, y, z, color='black', linewidth=3, alpha=1.0)
    
    # Plot highlight layer
    for idx, row in highlight_gdf.iterrows():
        geom = row.geometry
        if geom.geom_type in ['Polygon', 'MultiPolygon']:
            if geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    x, y = poly.exterior.xy
                    z = np.zeros(len(x))
                    verts = [list(zip(x, y, z))]
                    poly3d = Poly3DCollection(verts,
                                            alpha=0.8,
                                            facecolor=row['highlight_color'],
                                            edgecolor='black',
                                            linewidth=1.0)
                    ax.add_collection3d(poly3d)
            else:
                x, y = geom.exterior.xy
                z = np.zeros(len(x))
                verts = [list(zip(x, y, z))]
                poly3d = Poly3DCollection(verts,
                                        alpha=0.8,
                                        facecolor=row['highlight_color'],
                                        edgecolor='black',
                                        linewidth=1.0)
                ax.add_collection3d(poly3d)
    
 
    
    # Save the combined map
    output_path = os.path.join(output_dir, f'combined_{map_name}.png')
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    
    log.info(f"Saved combined map {map_name} to {output_path}")
    return output_path

# %%
# Create combined maps
combined_maps = {
    'isolated_signal': (base_layer, isolated_tracts),
    'no_311': (base_layer, no_311_tracts),
    'no_floodnet': (base_layer, no_floodnet_tracts),
    'no_dep': (base_layer, no_dep_tracts),
}

for map_name, (base_gdf, highlight_gdf) in combined_maps.items():
    log.info(f"Creating combined map: {map_name}")
    create_combined_3d_map(base_gdf, highlight_gdf, map_name, output_dir)

log.info("All 3D coverage maps have been generated successfully!") 