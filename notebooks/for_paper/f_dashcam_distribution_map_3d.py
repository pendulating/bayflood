# %%
from glob import glob 
from tqdm import tqdm
import os
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import palettable
import pandas as pd 
import geopandas as gpd 
import geodatasets 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import constants as c
import helpers as h
from logger import setup_logger 

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=8)

log = setup_logger("paper-analysis-dashcam-distribution-map-3d")
log.setLevel("INFO")
log.info("Modules loaded successfully.")

# %%
h.latex(True)

# %%
md_paths = glob("/share/ju/nexar_data/2023/2023-09-29/*/metadata.csv")
md = pd.concat([pd.read_csv(p, engine='pyarrow') for p in tqdm(md_paths, desc='Reading september 29 metadata...')])

# %%
# now, load images via checking if they exist from path in md 
img_paths_from_md = md['h3_index_res06'].astype(str).map(lambda x: os.path.join(
    "/share/ju/nexar_data/2023/2023-09-29",
    x,
    "frames/"
)) + md['frame_id'].astype(str) + ".jpg"
md['downloaded'] = img_paths_from_md.parallel_apply(lambda p: os.path.exists(p))
md['downloaded'].value_counts()

# %%
md = md[md['downloaded']]
md = gpd.GeoDataFrame(md, geometry=gpd.points_from_xy(md['gps_info.longitude'], md['gps_info.latitude'], crs=c.WGS)).to_crs(c.PROJ)

# %%
ct_nyc = gpd.read_file(f"{c.GEO_PATH}/ct-nyc-2020.geojson", crs=c.WGS).to_crs(c.PROJ)
nybb = gpd.read_file(geodatasets.get_path('nybb'), crs=c.WGS).to_crs(c.PROJ)

# %%
# count frames in each ct 
ct_counts = ct_nyc.sindex.query(md.geometry, predicate='intersects')[1,:]

# group by unique indexes
ct_counts = pd.Series(ct_counts).value_counts().sort_index()
# merge on index back into ct_nyc 
ct_nyc = ct_nyc.merge(ct_counts, left_index=True, right_index=True).rename(columns={'count':'nframes'})

# %%
# log the number of areas with 0 frames 
log.info(f"Number of areas with 0 frames: {ct_nyc['nframes'].isna().sum()}")
# log the smallest non-zero positive number of frames in an area 
log.info(f"Smallest non-zero number of frames: {ct_nyc[ct_nyc['nframes'] > 0]['nframes'].min()}")

# log the distribution of frames 
log.info(f"Frames distribution: \n{ct_nyc['nframes'].describe([0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99]).to_string()}")

# %%
# how many tracts have fewer than 50 frames 
log.info(f"Number of tracts with fewer than 50 frames: {ct_nyc[ct_nyc['nframes'] < 50].shape[0]}")
# fraction of tracts with fewer than 50 frames
log.info(f"Fraction of tracts with fewer than 50 frames: {ct_nyc[ct_nyc['nframes'] < 50].shape[0] / ct_nyc.shape[0]}")

# %%
def format_range(lower, upper):
    """Format a range of numbers with k/M suffixes"""
    def format_single(x):
        if x < 1000:
            return str(int(x))
        elif x < 1000000:
            return f'{int(x/1000)}k'
        else:
            return f'{int(x/1000000)}M'
    return f'{format_single(lower)}-{format_single(upper)}'

def create_intermediate_bins(start, end, base=10):
    """Create intermediate bins between powers of base"""
    start_exp = np.ceil(np.log10(start))
    end_exp = np.floor(np.log10(end))
    
    bins = [start]
    
    for exp in np.arange(start_exp, end_exp + 1):
        main_value = base ** exp
        bins.append(main_value)
        
        intermediate = 5 * main_value
        if intermediate <= end:
            bins.append(intermediate)
    
    if bins[-1] != end:
        bins.append(end)
    
    return np.array(bins)

def create_3d_dashcam_distribution_map(ct_nyc, nybb, output_dir):
    """
    Create a 3D version of the dashcam distribution map
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the bounds from nybb
    bounds = nybb.total_bounds
    
    # Get data range
    data_min = ct_nyc['nframes'].min()
    data_max = ct_nyc['nframes'].max()
    smallest_nonzero = ct_nyc['nframes'].value_counts().sort_index().iloc[0:].head(1).index[0]
    
    # Create bins and colormap
    bins = create_intermediate_bins(smallest_nonzero, 50000)
    # combine the last and second to last bin 
    bins[-2] = bins[-1]
    # combine the first and second bin 
    bins[1] = bins[0]
    bins = bins[0], *bins[2:-1]
    print(bins)
    
    cmap = matplotlib.cm.get_cmap('cividis')
    norm = BoundaryNorm(bins, cmap.N)
    
    # 1. Create the main 3D map (base layer)
    fig_map = plt.figure(figsize=(20, 20), dpi=300)
    ax_map = fig_map.add_axes([0, 0, 1, 1], projection='3d')
    
    # Set consistent view parameters
    ax_map.view_init(elev=90, azim=270)
    ax_map.set_proj_type('persp')
    
    # Set consistent bounds
    ax_map.set_xlim(bounds[0], bounds[2])
    ax_map.set_ylim(bounds[1], bounds[3])
    ax_map.set_zlim(0, 1)
    
    # Remove all padding and margins
    ax_map.margins(0)
    ax_map.set_axis_off()
    
    # Plot census tracts with nframes coloring
    for idx, row in ct_nyc.iterrows():
        geom = row.geometry
        if geom.geom_type in ['Polygon', 'MultiPolygon']:
            if geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    x, y = poly.exterior.xy
                    z = np.zeros(len(x))
                    verts = [list(zip(x, y, z))]
                    color = cmap(norm(row['nframes']))
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
                color = cmap(norm(row['nframes']))
                poly3d = Poly3DCollection(verts,
                                        alpha=0.8,
                                        facecolor=color,
                                        edgecolor='black',
                                        linewidth=0.2)
                ax_map.add_collection3d(poly3d)
    
    # Plot NYC borough boundary
    for idx, row in nybb.iterrows():
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
    
    # Save the base map
    base_map_output_path = os.path.join(output_dir, 'dashcam_distribution_3d_base.pdf')
    plt.savefig(base_map_output_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig_map)
    
    # 2. Create separate colorbar
    fig_cbar = plt.figure(figsize=(8, 2), dpi=300)
    cax = fig_cbar.add_axes([0.1, 0.2, 0.8, 0.6])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
    
    # Calculate tick positions and set colorbar properties
    tick_positions = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
    cbar.set_ticks(tick_positions)
    
    # Create range labels
    tick_labels = [format_range(bins[i], bins[i+1]) for i in range(len(bins)-1)]
    cbar.ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=24)
    
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.set_label('Number of Dashcam Images', fontsize=34, labelpad=10)
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.xaxis.set_tick_params(labelsize=32)

    
    # Save the colorbar
    cbar_output_path = os.path.join(output_dir, 'dashcam_distribution_3d_colorbar.pdf')
    plt.savefig(cbar_output_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig_cbar)
    
    # 3. Create separate histogram
    fig_hist = plt.figure(figsize=(6, 4), dpi=300)
    ax_hist = fig_hist.add_axes([0.1, 0.1, 0.8, 0.8])
    
    # Calculate and plot histogram bars
    bar_width = 1 / (len(bins)-1)  # Make bars thinner than color segments
    bin_counts = []
    for i in range(len(bins)-1):
        bin_mask = (ct_nyc['nframes'] >= bins[i]) & (ct_nyc['nframes'] < bins[i+1])
        count = np.sum(bin_mask)
        bin_counts.append(count)
        color = cmap(norm(bins[i]))
        
        # Calculate the center position
        x_pos = (i + 0.5)/(len(bins)-1)  # This will align with colorbar segments
        
        ax_hist.bar(x_pos,  # Center in color segment
                   count,
                   width=bar_width,
                   color=color,
                   edgecolor='black',
                   linewidth=0.5)
    
    # Set the x-axis limits to match colorbar
    ax_hist.set_xlim(0, 1)
    
    # Configure histogram axis
    ax_hist.spines['top'].set_visible(True)
    ax_hist.spines['right'].set_visible(True)
    ax_hist.spines['bottom'].set_visible(True)
    ax_hist.spines['left'].set_visible(True)
    
    # Calculate tick positions at center of each color block
    tick_positions = [(i + 0.5)/(len(bins)-1) for i in range(len(bins)-1)]
    ax_hist.set_xticks(tick_positions)
    
    
    # Style the ticks
    ax_hist.xaxis.set_ticks_position('bottom')
    ax_hist.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=24)
    # label the x-axis
    ax_hist.set_xlabel('Number of Dashcam Image in Tract', fontsize=32)
    # label the y-axis
    ax_hist.set_ylabel('Number of Tracts', fontsize=32)
    
    ax_hist.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Add more frequent y ticks
    max_count = max(bin_counts)
    ax_hist.set_yticks([0, 250, 500, 750, 1000, 1250, 1500])
    ax_hist.yaxis.set_tick_params(labelsize=22)
    
    # Save the histogram
    hist_output_path = os.path.join(output_dir, 'dashcam_distribution_3d_histogram.pdf')
    plt.savefig(hist_output_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig_hist)
    
    log.info(f"Saved base map to {base_map_output_path}")
    log.info(f"Saved colorbar to {cbar_output_path}")
    log.info(f"Saved histogram to {hist_output_path}")
    
    return base_map_output_path, cbar_output_path, hist_output_path

# %%
# Create the 3D dashcam distribution map
output_dir = f"{c.PAPER_PATH}/figures/3d_dashcam_distribution"

base_path, cbar_path, hist_path = create_3d_dashcam_distribution_map(ct_nyc, nybb, output_dir)

log.info("3D dashcam distribution map, colorbar, and histogram have been generated successfully!") 