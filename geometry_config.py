"""
Geometry configuration module for multi-geometry support.

This module defines the geometry types and provides path/configuration
factories for working with different census geography levels (Census Tracts,
Census Block Groups, Census Blocks).
"""

from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os


class GeometryType(Enum):
    """Supported census geometry types."""
    CT = "ct"      # Census Tracts
    CBG = "cbg"    # Census Block Groups
    CB = "cb"      # Census Blocks


@dataclass
class GeometryConfig:
    """Configuration for a specific geometry type."""
    
    geometry_type: GeometryType
    display_name: str
    id_column: str  # Column name for the geometry ID (GEOID)
    
    # File naming components
    file_prefix: str  # e.g., "ct", "cbg", "cb"
    
    # Default buffer distance for adjacency (in feet, for EPSG:2263)
    default_adjacency_buffer: float = 500.0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.geometry_type.value != self.file_prefix:
            raise ValueError(
                f"file_prefix '{self.file_prefix}' should match "
                f"geometry_type value '{self.geometry_type.value}'"
            )


# Geometry configurations registry
GEOMETRY_CONFIGS = {
    GeometryType.CT: GeometryConfig(
        geometry_type=GeometryType.CT,
        display_name="Census Tract",
        id_column="GEOID",
        file_prefix="ct",
        default_adjacency_buffer=500.0,
    ),
    GeometryType.CBG: GeometryConfig(
        geometry_type=GeometryType.CBG,
        display_name="Census Block Group",
        id_column="GEOID",
        file_prefix="cbg",
        default_adjacency_buffer=300.0,  # Smaller buffer for finer geometry
    ),
    GeometryType.CB: GeometryConfig(
        geometry_type=GeometryType.CB,
        display_name="Census Block",
        id_column="GEOID20",  # Census blocks use GEOID20 in 2020 shapefiles
        file_prefix="cb",
        default_adjacency_buffer=100.0,  # Even smaller for census blocks
    ),
}


def get_geometry_config(geometry_type: GeometryType) -> GeometryConfig:
    """Get configuration for a specific geometry type."""
    if geometry_type not in GEOMETRY_CONFIGS:
        raise ValueError(f"Unknown geometry type: {geometry_type}")
    return GEOMETRY_CONFIGS[geometry_type]


class GeometryPaths:
    """
    Path factory for geometry-specific file paths.
    
    Centralizes all path logic for different geometry types to ensure
    consistent naming conventions across the codebase.
    """
    
    def __init__(
        self, 
        geometry_type: GeometryType,
        base_dir: Optional[str] = None
    ):
        """
        Initialize path factory.
        
        Parameters
        ----------
        geometry_type : GeometryType
            The census geometry type
        base_dir : str, optional
            Base directory for all paths. Defaults to the bayflood repo root.
        """
        self.geometry_type = geometry_type
        self.config = get_geometry_config(geometry_type)
        self.base_dir = Path(base_dir) if base_dir else self._get_default_base_dir()
        
    @staticmethod
    def _get_default_base_dir() -> Path:
        """Get the default base directory (repo root)."""
        # Try to find the repo root by looking for config.py
        current = Path(__file__).parent
        if (current / "config.py").exists():
            return current
        # Fallback to environment variable or current directory
        return Path(os.getenv("BAYFLOOD_ROOT", "."))
    
    @property
    def prefix(self) -> str:
        """Get the file prefix for this geometry type."""
        return self.config.file_prefix
    
    # -------------------------------------------------------------------------
    # GeoJSON paths
    # -------------------------------------------------------------------------
    
    @property
    def geojson_dir(self) -> Path:
        """Directory containing geojson files."""
        return self.base_dir / "data"
    
    @property
    def geojson_path(self) -> Path:
        """Path to the main geojson file for this geometry."""
        return self.geojson_dir / f"{self.prefix}-nyc-2020.geojson"
    
    @property
    def geojson_with_water_path(self) -> Path:
        """Path to geojson including water areas."""
        return self.geojson_dir / f"{self.prefix}-nyc-wi-2020.geojson"
    
    @property
    def aggregation_geojson_path(self) -> Path:
        """Path to geojson in aggregation directory."""
        return self.base_dir / "aggregation" / "geo" / "data" / f"{self.prefix}-nyc-2020.geojson"
    
    # -------------------------------------------------------------------------
    # Adjacency paths
    # -------------------------------------------------------------------------
    
    @property
    def adjacency_dir(self) -> Path:
        """Directory for adjacency files."""
        buffer = int(self.config.default_adjacency_buffer)
        
        # For CT, check for legacy path structure first (cg_500 without ct_ prefix)
        if self.geometry_type == GeometryType.CT:
            legacy_dir = self.base_dir / "data" / "adjacency" / f"cg_{buffer}"
            if legacy_dir.exists():
                return legacy_dir
        
        # New consistent naming: {prefix}_cg_{buffer}
        return self.base_dir / "data" / "adjacency" / f"{self.prefix}_cg_{buffer}"
    
    def adjacency_node1_path(self, method: str = "custom_geometric") -> Path:
        """Path to adjacency node1 file."""
        return self.adjacency_dir / f"{self.prefix}_nyc_adj_list_{method}_node1.txt"
    
    def adjacency_node2_path(self, method: str = "custom_geometric") -> Path:
        """Path to adjacency node2 file."""
        return self.adjacency_dir / f"{self.prefix}_nyc_adj_list_{method}_node2.txt"
    
    def adjacency_matrix_path(self, method: str = "custom_geometric") -> Path:
        """Path to adjacency matrix (npy) file."""
        return self.adjacency_dir / f"{self.prefix}_nyc_adj_matrix_{method}.npy"
    
    # -------------------------------------------------------------------------
    # Dataset paths
    # -------------------------------------------------------------------------
    
    @property
    def processed_data_dir(self) -> Path:
        """Directory for processed datasets."""
        return self.base_dir / "data" / "processed"
    
    @property
    def flooding_dataset_path(self) -> Path:
        """Path to the main flooding dataset for this geometry."""
        return self.processed_data_dir / f"flooding_{self.prefix}_dataset.csv"
    
    @property
    def aggregation_dir(self) -> Path:
        """Directory for aggregation outputs."""
        return self.base_dir / "aggregation"
    
    def context_df_path(self, date_str: str) -> Path:
        """
        Path to context dataframe for a specific date.
        
        Parameters
        ----------
        date_str : str
            Date string in MMDDYYYY format
        """
        return self.aggregation_dir / f"context_df_{self.prefix}_{date_str}.csv"
    
    def context_df_describe_path(self, date_str: str) -> Path:
        """Path to context dataframe description for a specific date."""
        return self.aggregation_dir / f"context_df_{self.prefix}_describe_{date_str}.csv"
    
    # -------------------------------------------------------------------------
    # Topology paths
    # -------------------------------------------------------------------------
    
    @property
    def topology_dir(self) -> Path:
        """Directory for topology data."""
        return self.base_dir / "aggregation" / "geo" / "data" / "processed"
    
    @property
    def topology_path(self) -> Path:
        """Path to topology CSV for this geometry."""
        return self.topology_dir / f"{self.prefix}_nyc_topology.csv"
    
    # -------------------------------------------------------------------------
    # Run output paths
    # -------------------------------------------------------------------------
    
    def run_dir(self, run_id: str) -> Path:
        """Get the run directory for a specific run ID."""
        return self.base_dir / "runs" / run_id
    
    def estimate_path(self, run_id: str, estimate_name: str) -> Path:
        """Path to estimate CSV for a specific run."""
        return self.run_dir(run_id) / f"estimate_{estimate_name}.csv"
    
    def analysis_df_path(self, run_id: str, prefix: str, date_str: str) -> Path:
        """Path to analysis dataframe for a specific run."""
        return self.run_dir(run_id) / f"analysis_df_{prefix}_{date_str}.csv"


def get_geometry_paths(
    geometry_type: GeometryType | str,
    base_dir: Optional[str] = None
) -> GeometryPaths:
    """
    Factory function to get a GeometryPaths instance.
    
    Parameters
    ----------
    geometry_type : GeometryType or str
        The geometry type (can be enum or string like "ct", "cbg", "cb")
    base_dir : str, optional
        Base directory for paths
        
    Returns
    -------
    GeometryPaths
        Path factory for the specified geometry type
    """
    if isinstance(geometry_type, str):
        geometry_type = GeometryType(geometry_type.lower())
    return GeometryPaths(geometry_type, base_dir)


# Convenience functions for common path lookups
def get_geojson_path(geometry_type: GeometryType | str, base_dir: Optional[str] = None) -> Path:
    """Get the geojson path for a geometry type."""
    return get_geometry_paths(geometry_type, base_dir).geojson_path


def get_flooding_dataset_path(geometry_type: GeometryType | str, base_dir: Optional[str] = None) -> Path:
    """Get the flooding dataset path for a geometry type."""
    return get_geometry_paths(geometry_type, base_dir).flooding_dataset_path


def get_adjacency_paths(
    geometry_type: GeometryType | str,
    base_dir: Optional[str] = None,
    method: str = "custom_geometric"
) -> tuple[Path, Path]:
    """
    Get adjacency node1 and node2 paths for a geometry type.
    
    Returns
    -------
    tuple[Path, Path]
        (node1_path, node2_path)
    """
    paths = get_geometry_paths(geometry_type, base_dir)
    return paths.adjacency_node1_path(method), paths.adjacency_node2_path(method)


# Default geometry type - can be overridden by environment variable
DEFAULT_GEOMETRY_TYPE = GeometryType(
    os.getenv("BAYFLOOD_GEOMETRY_TYPE", "ct").lower()
)


if __name__ == "__main__":
    # Quick test of the module
    for geo_type in GeometryType:
        print(f"\n=== {geo_type.name} ({geo_type.value}) ===")
        paths = get_geometry_paths(geo_type)
        print(f"  Display name: {paths.config.display_name}")
        print(f"  ID column: {paths.config.id_column}")
        print(f"  GeoJSON: {paths.geojson_path}")
        print(f"  Flooding dataset: {paths.flooding_dataset_path}")
        print(f"  Adjacency dir: {paths.adjacency_dir}")
        node1, node2 = get_adjacency_paths(geo_type)
        print(f"  Adjacency node1: {node1}")
        print(f"  Adjacency node2: {node2}")

