"""
Constants and path configuration for for_revisions notebooks.

Supports multiple census geometry types (CT, CBG, CB).
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from geometry_config import GeometryType, get_geometry_paths, get_geometry_config

BASE_DIR = Path(__file__).parent.parent.parent

# -----------------------------------------------------------------------------
# Current Run Paths (Census Tract by default)
# -----------------------------------------------------------------------------

CURRENT_DF = str(BASE_DIR / "runs/icar_icar/simulated_False/ahl_True/covariates_True/FINAL_20260206-1100/analysis_df_FINAL_02062026.csv")

CURRENT_NO_COVARIATES_DF = str(BASE_DIR / "runs/icar_icar/simulated_False/ahl_True/covariates_False/FINAL_20260206-1205/analysis_df_FINAL_02062026.csv")

# Output paths
PAPER_PATH = str(BASE_DIR / 'papers/natcities_bayflood_2025')
DELIVERABLES_PATH = str(BASE_DIR / 'deliverables')

# Coordinate reference systems
WGS='EPSG:4326'
PROJ='EPSG:2263'

# Legacy paths (Census Tract specific)
GEO_PATH='../../aggregation/geo/data'

CURRENT_PP_BASELINES_GLOB=""

CURRENT_ADJ_1 = str(BASE_DIR / "data/adjacency/cg_500/ct_nyc_adj_list_custom_geometric_node1.txt")
CURRENT_ADJ_2 = str(BASE_DIR / "data/adjacency/cg_500/ct_nyc_adj_list_custom_geometric_node2.txt")

ESTIMATE_TO_USE='confirmed_or_above_thres'

# -----------------------------------------------------------------------------
# Geometry-Aware Path Functions
# -----------------------------------------------------------------------------


def get_geojson_path(geometry_type: GeometryType | str = GeometryType.CT) -> Path:
    """Get the geojson path for a geometry type."""
    paths = get_geometry_paths(geometry_type, str(BASE_DIR))
    geojson_path = paths.aggregation_geojson_path
    if not geojson_path.exists():
        geojson_path = paths.geojson_path
    return geojson_path


def get_flooding_dataset_path(geometry_type: GeometryType | str = GeometryType.CT) -> Path:
    """Get the flooding dataset path for a geometry type."""
    return get_geometry_paths(geometry_type, str(BASE_DIR)).flooding_dataset_path


def get_adjacency_paths(
    geometry_type: GeometryType | str = GeometryType.CT,
    method: str = "custom_geometric"
) -> tuple[Path, Path]:
    """
    Get adjacency node1 and node2 paths for a geometry type.
    
    Returns
    -------
    tuple[Path, Path]
        (node1_path, node2_path)
    """
    paths = get_geometry_paths(geometry_type, str(BASE_DIR))
    return paths.adjacency_node1_path(method), paths.adjacency_node2_path(method)


def get_topology_path(geometry_type: GeometryType | str = GeometryType.CT) -> Path:
    """Get the topology data path for a geometry type."""
    return get_geometry_paths(geometry_type, str(BASE_DIR)).topology_path


def get_context_df_path(
    geometry_type: GeometryType | str = GeometryType.CT,
    date: str = "12012025"
) -> Path:
    """Get the context dataframe path for a geometry type and date."""
    paths = get_geometry_paths(geometry_type, str(BASE_DIR))
    prefix = paths.config.prefix
    return paths.aggregation_dir / f"context_df_{prefix}_{date}.csv"


def get_id_column(geometry_type: GeometryType | str = GeometryType.CT) -> str:
    """Get the ID column name for a geometry type."""
    if isinstance(geometry_type, str):
        geometry_type = GeometryType(geometry_type.lower())
    return get_geometry_config(geometry_type).id_column


def get_display_name(geometry_type: GeometryType | str = GeometryType.CT) -> str:
    """Get the display name for a geometry type (e.g., 'Census Tract')."""
    if isinstance(geometry_type, str):
        geometry_type = GeometryType(geometry_type.lower())
    return get_geometry_config(geometry_type).display_name


# -----------------------------------------------------------------------------
# Convenience: Geometry-specific path dictionaries
# -----------------------------------------------------------------------------

CT_PATHS = {
    'geojson': get_geojson_path(GeometryType.CT),
    'flooding_dataset': get_flooding_dataset_path(GeometryType.CT),
    'adjacency': get_adjacency_paths(GeometryType.CT),
    'topology': get_topology_path(GeometryType.CT),
}

CBG_PATHS = {
    'geojson': get_geojson_path(GeometryType.CBG),
    'flooding_dataset': get_flooding_dataset_path(GeometryType.CBG),
    'adjacency': get_adjacency_paths(GeometryType.CBG),
    'topology': get_topology_path(GeometryType.CBG),
}

CB_PATHS = {
    'geojson': get_geojson_path(GeometryType.CB),
    'flooding_dataset': get_flooding_dataset_path(GeometryType.CB),
    'adjacency': get_adjacency_paths(GeometryType.CB),
    'topology': get_topology_path(GeometryType.CB),
}
