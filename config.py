"""
Configuration module for bayflood pipeline.

Provides configuration values that can be overridden via environment variables.
Supports multi-geometry configurations (Census Tracts, Block Groups, Blocks).
"""

import os
from pathlib import Path

from geometry_config import (
    GeometryType,
    GeometryPaths,
    get_geometry_paths,
    get_geometry_config,
    DEFAULT_GEOMETRY_TYPE,
)


def _as_bool(value: str, default: bool = False) -> bool:
    """Convert string environment variable to boolean."""
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


# -----------------------------------------------------------------------------
# Geometry Type Configuration
# -----------------------------------------------------------------------------

# Current geometry type - can be set via BAYFLOOD_GEOMETRY_TYPE env var
GEOMETRY_TYPE: GeometryType = DEFAULT_GEOMETRY_TYPE

# Get paths helper for current geometry
_geometry_paths = get_geometry_paths(GEOMETRY_TYPE)


def get_paths_for_geometry(geometry_type: GeometryType | str | None = None) -> GeometryPaths:
    """
    Get GeometryPaths instance for a specific geometry type.
    
    Parameters
    ----------
    geometry_type : GeometryType, str, or None
        The geometry type. If None, uses the default GEOMETRY_TYPE.
        
    Returns
    -------
    GeometryPaths
        Path factory for the specified geometry type
    """
    if geometry_type is None:
        geometry_type = GEOMETRY_TYPE
    return get_geometry_paths(geometry_type)


# -----------------------------------------------------------------------------
# Dataset Paths (with backward compatibility)
# -----------------------------------------------------------------------------

def _get_default_dataset_path() -> str:
    """Get the default dataset path based on geometry type."""
    # First check for explicit environment variable (backward compatible)
    env_path = os.getenv("EMPIRICAL_DATA_PATH")
    if env_path:
        return env_path
    
    # Use geometry-specific flooding dataset path
    return str(_geometry_paths.flooding_dataset_path)


def _get_default_adj_node1_path() -> str:
    """Get the default adjacency node1 path based on geometry type."""
    env_path = os.getenv("ADJ_NODE1_PATH")
    if env_path:
        return env_path
    return str(_geometry_paths.adjacency_node1_path())


def _get_default_adj_node2_path() -> str:
    """Get the default adjacency node2 path based on geometry type."""
    env_path = os.getenv("ADJ_NODE2_PATH")
    if env_path:
        return env_path
    return str(_geometry_paths.adjacency_node2_path())


# Core dataset and adjacency defaults
# These maintain backward compatibility with existing code that imports them directly
DATASET_PATH: str = _get_default_dataset_path()

ADJ_NODE1_PATH: str = _get_default_adj_node1_path()
ADJ_NODE2_PATH: str = _get_default_adj_node2_path()
ADJ_NPY_PATH: str | None = os.getenv("ADJ_NPY_PATH")


# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------

EXTERNAL_COVARIATES: bool = _as_bool(os.getenv("EXTERNAL_COVARIATES"), default=False)

# Optional: default sampling settings (can still be overridden at CLI)
DEFAULT_WARMUP: int = int(os.getenv("DEFAULT_WARMUP", "1000"))
DEFAULT_SAMPLES: int = int(os.getenv("DEFAULT_SAMPLES", "1500"))


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------

def get_geojson_path(geometry_type: GeometryType | str | None = None) -> Path:
    """Get the geojson path for a geometry type."""
    return get_paths_for_geometry(geometry_type).geojson_path


def get_flooding_dataset_path(geometry_type: GeometryType | str | None = None) -> Path:
    """Get the flooding dataset path for a geometry type."""
    return get_paths_for_geometry(geometry_type).flooding_dataset_path


def get_adjacency_paths(
    geometry_type: GeometryType | str | None = None,
    method: str = "custom_geometric"
) -> tuple[Path, Path]:
    """
    Get adjacency node1 and node2 paths for a geometry type.
    
    Returns
    -------
    tuple[Path, Path]
        (node1_path, node2_path)
    """
    paths = get_paths_for_geometry(geometry_type)
    return paths.adjacency_node1_path(method), paths.adjacency_node2_path(method)


def get_id_column(geometry_type: GeometryType | str | None = None) -> str:
    """Get the ID column name for a geometry type (e.g., 'GEOID')."""
    if geometry_type is None:
        geometry_type = GEOMETRY_TYPE
    if isinstance(geometry_type, str):
        geometry_type = GeometryType(geometry_type.lower())
    return get_geometry_config(geometry_type).id_column


# -----------------------------------------------------------------------------
# Legacy compatibility - maintain old variable names pointing to CT-specific paths
# These are deprecated and should be replaced with geometry-aware functions
# -----------------------------------------------------------------------------

# For backward compatibility with code that still uses the old cg_500 path structure
_LEGACY_ADJ_DIR = Path("data/adjacency/cg_500")
LEGACY_ADJ_NODE1_PATH = _LEGACY_ADJ_DIR / "ct_nyc_adj_list_custom_geometric_node1.txt"
LEGACY_ADJ_NODE2_PATH = _LEGACY_ADJ_DIR / "ct_nyc_adj_list_custom_geometric_node2.txt"
