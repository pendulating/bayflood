# %%
import pandas as pd
import geopandas as gpd 
import geodatasets
import numpy as np
from typing import Dict, Set, List
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
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

log = setup_logger("floodnet-sensor-placement-algorithm-3d")
log.setLevel("INFO")
log.info("Modules loaded.")

h.latex(True)

# %%
ct_nyc = gpd.read_file(f"{c.GEO_PATH}/ct-nyc-2020.geojson", crs=c.WGS).to_crs(c.PROJ)
nybb = gpd.read_file(geodatasets.get_path("nybb"), crs=c.WGS).to_crs(c.PROJ)
log.info("Loaded NYC Census Tracts and Boroughs.")

# %%
def optimize_sensor_placement(
    df: pd.DataFrame,
    n_new_sensors: int,
    adjacency_map: Dict[int, Set[int]],
    r: int = 1
    ) -> List[int]:
    working_df = df.copy()
    selected_tracts = []
    total_py_covered = 0

    def get_coverage(tract_id: int, radius: int) -> Set[int]:
        covered = {tract_id}
        frontier = {tract_id}
        
        for step in range(radius):
            new_frontier = set()
            for t in frontier:
                new_frontier.update(adjacency_map.get(t, set()))
            covered.update(new_frontier)
            frontier = new_frontier
            log.debug(f"Step {step+1}: Added {len(new_frontier)} tracts to coverage")
            
        return covered
   
    def score_placement(tract_id: int) -> float:
        if working_df.loc[working_df['BoroCT2020'] == tract_id, 'n_floodnet_sensors'].iloc[0] > 0:
            return -np.inf
            
        covered = get_coverage(tract_id, r)
        covered_tracts = working_df[working_df['BoroCT2020'].isin(covered)]
        
        score = covered_tracts[
            covered_tracts['n_floodnet_sensors'] == 0
        ]['p_y'].sum()
        
        log.debug(f"Tract {tract_id} would cover {len(covered)} tracts with score {score:.4f}")
        return score

    log.info(f"Starting placement of {n_new_sensors} sensors with radius {r}")
    log.info(f"Initial p_y coverage: {working_df[working_df['n_floodnet_sensors'] > 0]['p_y'].sum():.4f}")
    
    for i in range(n_new_sensors):
        scores = []
        for tract_id in working_df['BoroCT2020']:
            scores.append((tract_id, score_placement(tract_id)))
            
        best_tract, best_score = max(scores, key=lambda x: x[1])
        
        if best_score == -np.inf:
            log.info("No more valid placements available")
            break
            
        selected_tracts.append(best_tract)
        
        covered = get_coverage(best_tract, r)
        working_df.loc[
            working_df['BoroCT2020'].isin(covered),
            'n_floodnet_sensors'
        ] = 1
        
        new_total = working_df[working_df['n_floodnet_sensors'] > 0]['p_y'].sum()
        py_added = new_total - total_py_covered
        total_py_covered = new_total
        
        log.info(f"Placed sensor {i+1} at tract {best_tract}")
        log.info(f"Added p_y coverage: {py_added:.4f}")
        log.info(f"Total p_y coverage: {total_py_covered:.4f}")
        log.info(f"Newly covered tracts: {len(covered)}")
        
    log.info(f"Final p_y coverage: {total_py_covered:.4f}")
    log.info(f"Selected tracts: {selected_tracts}")
    
    return selected_tracts

# %%
analysis_df = pd.read_csv(c.CURRENT_DF)
analysis_df['og_sensor_count'] = analysis_df['n_floodnet_sensors']

# %%
def build_adjacency_map(node1_file: str, node2_file: str, df: pd.DataFrame) -> Dict[int, Set[int]]:
    # Read node files
    with open(node1_file) as f1, open(node2_file) as f2:
        node1 = [int(x) for x in f1.readlines()]
        node2 = [int(x) for x in f2.readlines()]
    
    # Create mapping from index to tract ID
    tract_ids = df['BoroCT2020']
    idx_to_tract = {i+1: tract for i, tract in enumerate(tract_ids)}
    
    # Build adjacency map using actual tract IDs
    adj_map = {}
    for n1, n2 in zip(node1, node2):
        tract1 = idx_to_tract[n1]
        tract2 = idx_to_tract[n2]
        
        if tract1 not in adj_map:
            adj_map[tract1] = set()
        if tract2 not in adj_map:
            adj_map[tract2] = set()
            
        adj_map[tract1].add(tract2)
        adj_map[tract2].add(tract1)
    
    return adj_map

# Use this to create the adjacency map
adj_map = build_adjacency_map(c.CURRENT_ADJ_1, c.CURRENT_ADJ_2, analysis_df)

# %%
new_sensor_locations = optimize_sensor_placement(analysis_df, 25, adj_map, r=1)

# %%
ct_nyc['selected'] = 0
ct_nyc['selected_order'] = 0
ct_nyc['BoroCT2020'] = ct_nyc['BoroCT2020'].astype(int)

for i, tract in enumerate(new_sensor_locations):
    ct_nyc.loc[ct_nyc['BoroCT2020'] == tract, 'selected'] = 1
    ct_nyc.loc[ct_nyc['BoroCT2020'] == tract, 'selected_order'] = i + 1

# %%
# add p_y to ct_nyc 
ct_nyc['p_y'] = analysis_df['p_y']

# %%
ct_nyc['n_added_sensors'] = 0
ct_nyc.loc[ct_nyc['selected'] == 1, 'n_added_sensors'] = 1

# %%
# load floodnet geo 
floodnet_sensors_sep23 = pd.read_csv("/share/ju/matt/street-flooding/aggregation/flooding/static/sep23_floodnet_sensor_coordinates.csv")
floodnet_sensors_sep23 = gpd.GeoDataFrame(floodnet_sensors_sep23, geometry=gpd.points_from_xy(floodnet_sensors_sep23['lon'], floodnet_sensors_sep23['lat']), crs=c.WGS).to_crs(c.PROJ)

floodnet_sensors_current = pd.read_csv("/share/ju/matt/street-flooding/aggregation/flooding/static/current_floodnet_sensors.csv")
floodnet_sensors_current = gpd.GeoDataFrame(floodnet_sensors_current, geometry=gpd.points_from_xy(floodnet_sensors_current['longitude'], floodnet_sensors_current['latitude']), crs=c.WGS).to_crs(c.PROJ)

# drop a sensor from floodnet_sensors_current if it is within THRES Ft of another 
log.info(f"Current floodnet sensors: {len(floodnet_sensors_current)}")
THRES = 10
for idx, row in floodnet_sensors_current.iterrows():
    for idx2, row2 in floodnet_sensors_current.iterrows():
        if idx == idx2:
            continue 
        if row['geometry'].distance(row2['geometry']) < THRES:
            floodnet_sensors_current.drop(idx, inplace=True)
            log.info(f"Dropped sensor {idx} from current floodnet sensors")
            break 
log.info(f"Current floodnet sensors after filtering for duplicates: {len(floodnet_sensors_current)}")

# %%
def create_3d_floodnet_placement_map(ct_nyc, floodnet_sensors_current, selected_tracts, output_dir):
    """
    Create a 3D version of the FloodNet placement map with separate legend and colorbar files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure all layers are in the same CRS as nybb
    if ct_nyc.crs != nybb.crs:
        ct_nyc = ct_nyc.to_crs(nybb.crs)
    if floodnet_sensors_current.crs != nybb.crs:
        floodnet_sensors_current = floodnet_sensors_current.to_crs(nybb.crs)
    if selected_tracts.crs != nybb.crs:
        selected_tracts = selected_tracts.to_crs(nybb.crs)
    
    # Get the bounds from nybb
    bounds = nybb.total_bounds
    
    # Set up colormap for p_y
    lognorm = LogNorm(vmin=ct_nyc['p_y'].min(), vmax=ct_nyc['p_y'].max())
    cmap = mpl.cm.get_cmap('coolwarm')
    
    # 1. Create the main 3D map (without legend and colorbar)
    fig_map = plt.figure(figsize=(20, 20), dpi=300)
    ax_map = fig_map.add_axes([0, 0, 1, 1], projection='3d')
    
    # Set consistent view parameters
    ax_map.view_init(elev=20, azim=270)
    ax_map.set_proj_type('persp')
    
    # Set consistent bounds
    ax_map.set_xlim(bounds[0], bounds[2])
    ax_map.set_ylim(bounds[1], bounds[3])
    ax_map.set_zlim(0, 1)
    
    # Remove all padding and margins
    ax_map.margins(0)
    ax_map.set_axis_off()
    
    # Plot census tracts with p_y coloring
    for idx, row in ct_nyc.iterrows():
        geom = row.geometry
        if geom.geom_type in ['Polygon', 'MultiPolygon']:
            if geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    x, y = poly.exterior.xy
                    z = np.zeros(len(x))
                    verts = [list(zip(x, y, z))]
                    color = cmap(lognorm(row['p_y']))
                    poly3d = Poly3DCollection(verts,
                                            alpha=0.8,
                                            facecolor=color,
                                            edgecolor='black',
                                            linewidth=0.2)
                    ax_map.add_collection3d(poly3d)
            else:
                x, y = geom.exterior.xy
                z = np.zeros(len(x))
                verts = [list(zip(x, y, z))]
                color = cmap(lognorm(row['p_y']))
                poly3d = Poly3DCollection(verts,
                                        alpha=0.8,
                                        facecolor=color,
                                        edgecolor='black',
                                        linewidth=0.2)
                ax_map.add_collection3d(poly3d)
    
    # Plot current FloodNet sensors as points
    sensor_x = []
    sensor_y = []
    for idx, row in floodnet_sensors_current.iterrows():
        sensor_x.append(row.geometry.x)
        sensor_y.append(row.geometry.y)
    
    # Plot NYC borough boundary (simplified approach)
    # Ensure nybb is in the same CRS as the plot
    if nybb.crs != ct_nyc.crs:
        nybb_plot = nybb.to_crs(ct_nyc.crs)
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
                        ax_map.plot(x, y, z, color='black', linewidth=3, alpha=1.0)
            elif geom.geom_type == 'Polygon':
                x, y = geom.exterior.xy
                z = np.zeros(len(x))
                ax_map.plot(x, y, z, color='black', linewidth=3, alpha=1.0)
    
    # Prepare selected tracts centroids for scatter layer
    selected_tracts['centroid'] = selected_tracts.centroid
    selected_x = []
    selected_y = []
    for idx, row in selected_tracts.iterrows():
        selected_x.append(row['centroid'].x)
        selected_y.append(row['centroid'].y)
    
    # Prepare current sensor coordinates for scatter layer
    sensor_x = []
    sensor_y = []
    for idx, row in floodnet_sensors_current.iterrows():
        sensor_x.append(row.geometry.x)
        sensor_y.append(row.geometry.y)
    
    # Save the base map (polygons and borough boundary only)
    base_map_output_path = os.path.join(output_dir, 'floodnet_placement_3d_base.pdf')
    plt.savefig(base_map_output_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig_map)
    
    # Create scatter-only map
    fig_scatter = plt.figure(figsize=(20, 20), dpi=300)
    ax_scatter = fig_scatter.add_axes([0, 0, 1, 1], projection='3d')
    
    # Set consistent view parameters
    ax_scatter.view_init(elev=20, azim=270)
    ax_scatter.set_proj_type('persp')
    
    # Set consistent bounds
    ax_scatter.set_xlim(bounds[0], bounds[2])
    ax_scatter.set_ylim(bounds[1], bounds[3])
    ax_scatter.set_zlim(0, 1)
    
    # Remove all padding and margins
    ax_scatter.margins(0)
    ax_scatter.set_axis_off()
    
    # Plot selected tracts centroids only
    if selected_x:
        ax_scatter.scatter(selected_x, selected_y, np.zeros(len(selected_x)),
                          c='green', s=200, alpha=1, marker='P', 
                          facecolor='limegreen', edgecolor='black', linewidth=1)
    
    # Plot current FloodNet sensors as points only
    if sensor_x:
        ax_scatter.scatter(sensor_x, sensor_y, np.zeros(len(sensor_x)),
                          c='black', s=75, alpha=0.7, marker='D', 
                          facecolor='lightgrey', edgecolor='black', linewidth=1)
    
    # Save the scatter-only map
    scatter_output_path = os.path.join(output_dir, 'floodnet_placement_3d_scatter.pdf')
    plt.savefig(scatter_output_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig_scatter)
    
    # 2. Create separate colorbar
    fig_cbar = plt.figure(figsize=(2, 8), dpi=300)
    cax = fig_cbar.add_axes([0.2, 0.1, 0.6, 0.8])
    norm = mpl.colors.Normalize(vmin=ct_nyc['p_y'].min(), vmax=ct_nyc['p_y'].max())
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=lognorm)
    cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label(r"$r_c$ (log scale)", size=22, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=16)
    
    # Save the colorbar
    cbar_output_path = os.path.join(output_dir, 'floodnet_placement_3d_colorbar.pdf')
    plt.savefig(cbar_output_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig_cbar)
    
    # 3. Create separate legend
    fig_legend = plt.figure(figsize=(6, 3), dpi=300)
    ax_legend = fig_legend.add_axes([0, 0, 1, 1])
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.set_axis_off()
    
    # Create legend elements
    legend_elements = [
        plt.Line2D([0], [0], marker='D', color='black', label='Current sensor location',
                   markerfacecolor='none', markeredgewidth=2, markersize=10, linestyle='None'),
        plt.Line2D([0], [0], marker='P', color='green', label='Suggested sensor in tract',
                   markerfacecolor='none', markeredgewidth=2, markersize=10, linestyle='None'),
    ]
    
    # Add legend
    legend = ax_legend.legend(handles=legend_elements, 
                             loc='center', 
                             fontsize=16,
                             handletextpad=1.5,
                             labelspacing=1.5,
                             borderpad=0.5,
                             borderaxespad=0.5,
                             framealpha=1,
                             edgecolor='black')
    
    # Save the legend
    legend_output_path = os.path.join(output_dir, 'floodnet_placement_3d_legend.pdf')
    plt.savefig(legend_output_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig_legend)
    
    log.info(f"Saved base map (polygons) to {base_map_output_path}")
    log.info(f"Saved scatter points to {scatter_output_path}")
    log.info(f"Saved colorbar to {cbar_output_path}")
    log.info(f"Saved legend to {legend_output_path}")
    
    return base_map_output_path, scatter_output_path, cbar_output_path, legend_output_path

# %%
# Create the 3D FloodNet placement map
selected_tracts = ct_nyc[ct_nyc['selected_order'] > 0].copy()
output_dir = f"{c.PAPER_PATH}/figures/3d_floodnet_placement"

base_path, scatter_path, cbar_path, legend_path = create_3d_floodnet_placement_map(ct_nyc, floodnet_sensors_current, selected_tracts, output_dir)

log.info("3D FloodNet placement base map, scatter points, colorbar, and legend have been generated successfully!") 