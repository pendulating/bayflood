"""
Compute and analyze different types of spatial weights for census geographies.

This module provides tools for generating adjacency matrices and spatial weights
for census tracts, block groups, or other geographic units.
"""

import sys
from pathlib import Path

# Add parent directories for geometry_config import
_this_file = Path(__file__).resolve()
_project_root = _this_file.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import geopandas as gpd
import numpy as np
import libpysal
from typing import Dict, Tuple, List, Optional, Union
import networkx as nx
from dataclasses import dataclass

try:
    from geometry_config import GeometryType, get_geometry_config, get_geometry_paths
    GEOMETRY_CONFIG_AVAILABLE = True
except ImportError:
    GEOMETRY_CONFIG_AVAILABLE = False
    GeometryType = None


@dataclass
class WeightsAnalysis:
    """Container for weights analysis results."""
    weights_matrix: np.ndarray
    n_connections: int
    isolated_areas: List[int]  # Renamed from isolated_tracts for generality
    avg_connections: float
    description: str
    
    # Backward compatibility property
    @property
    def isolated_tracts(self) -> List[int]:
        """Alias for isolated_areas for backward compatibility."""
        return self.isolated_areas


def compute_custom_geometric_weights(
    gdf: gpd.GeoDataFrame,
    buffer_dist: float = 0.1,
    blacklist: Dict[int, List[int]] = None,
    debug: bool = True
) -> np.ndarray:
    """
    Compute weights matrix using buffered geometric intersection.
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame in state plane projection
    buffer_dist : float
        Buffer distance in feet
    blacklist : Dict[int, List[int]], optional
        Dictionary where keys are 1-indexed area indices and values are lists
        of 1-indexed area indices they should not connect to
    debug : bool
        Whether to print debug information
    """
    n = len(gdf)
    W = np.zeros((n, n))
    
    # Validate blacklist
    if blacklist:
        # Check that all indices exist
        max_idx = len(gdf)
        all_indices = set()
        for k, v in blacklist.items():
            all_indices.add(k)
            all_indices.update(v)
        
        invalid_indices = [idx for idx in all_indices if idx < 1 or idx > max_idx]
        if invalid_indices:
            raise ValueError(f"Invalid indices in blacklist: {invalid_indices}. Valid range is 1 to {max_idx}")
        
        # Convert to 0-indexed
        blacklist_zero = {k-1: [x-1 for x in v] for k, v in blacklist.items()}
        
        if debug:
            print("\nBlacklist validation:")
            print(f"Number of areas with blacklist entries: {len(blacklist)}")
            print("Blacklisted connections:")
            for k, v in blacklist.items():
                print(f"Area {k} blocked from: {v}")
    else:
        blacklist_zero = {}

    # Track blocked connections for debugging
    blocked_connections = []
    
    for i in range(n):
        area = gdf.iloc[i]
        buffered_geom = area.geometry.buffer(buffer_dist)
        neighbors_query = gdf.geometry.intersects(buffered_geom)
        neighbors = gdf[neighbors_query].index
        
        for j in neighbors:
            if i != j:
                # Check if this connection is blacklisted
                connection_blocked = (
                    i in blacklist_zero and j in blacklist_zero.get(i, []) or
                    j in blacklist_zero and i in blacklist_zero.get(j, [])
                )
                
                if connection_blocked:
                    blocked_connections.append((i+1, j+1))  # Store 1-indexed for output
                    if debug:
                        print(f"Blocking connection between areas {i+1} and {j+1}")
                else:
                    W[i,j] = 1
    
    if debug and blacklist:
        print(f"\nTotal connections blocked: {len(blocked_connections)}")
        # Verify symmetry
        asymmetric = []
        for i in range(n):
            for j in range(i+1, n):
                if W[i,j] != W[j,i]:
                    asymmetric.append((i+1, j+1))
        if asymmetric:
            print("Warning: Found asymmetric connections:", asymmetric)
    
    return W


def analyze_weights(W: np.ndarray) -> Tuple[int, List[int], float]:
    """Analyze a weights matrix for basic connectivity statistics."""
    G = nx.from_numpy_array(W)
    n_connections = int(W.sum() / 2)  # Divide by 2 for undirected graph
    isolated_areas = list(nx.isolates(G))
    avg_connections = W.sum() / len(W)
    return n_connections, isolated_areas, avg_connections


class GeometryWeightsGenerator:
    """
    Generate and analyze different types of spatial weights for census geographies.
    
    Works with any census geography level: tracts, block groups, or blocks.
    """
    
    def __init__(
        self, 
        shapefile_path_or_geometry_type: Union[str, Path, "GeometryType"],
        id_column: str = None,
        geometry_prefix: str = None
    ):
        """
        Initialize with shapefile path or GeometryType.
        
        Parameters:
        -----------
        shapefile_path_or_geometry_type : str, Path, or GeometryType
            Either a path to census geography shapefile/geojson, or a GeometryType
            enum value (CT, CBG, CB) which will auto-resolve paths.
        id_column : str, optional
            Column name containing the geography identifier.
            If None and GeometryType provided, uses config default.
        geometry_prefix : str, optional
            Prefix for output file names (e.g., "ct", "cbg", "cb").
            If None and GeometryType provided, uses geometry type value.
        """
        # Handle GeometryType input
        if GEOMETRY_CONFIG_AVAILABLE and isinstance(shapefile_path_or_geometry_type, GeometryType):
            geometry_type = shapefile_path_or_geometry_type
            paths = get_geometry_paths(geometry_type, str(_project_root))
            config = get_geometry_config(geometry_type)
            
            shapefile_path = paths.geojson_path
            if id_column is None:
                id_column = config.id_column
            if geometry_prefix is None:
                geometry_prefix = geometry_type.value
        else:
            shapefile_path = shapefile_path_or_geometry_type
            if id_column is None:
                id_column = "GEOID"
            if geometry_prefix is None:
                geometry_prefix = "geo"
        
        self.gdf = gpd.read_file(str(shapefile_path))
        self.id_column = id_column
        self.geometry_prefix = geometry_prefix
        
        # Try to convert ID column to int if possible (for backward compatibility)
        if id_column in self.gdf.columns:
            try:
                self.gdf[id_column] = self.gdf[id_column].astype(int)
            except (ValueError, TypeError):
                # Keep as string if conversion fails
                pass
        
        # Also check for BoroCT2020 for backward compatibility with CT data
        if 'BoroCT2020' in self.gdf.columns:
            try:
                self.gdf['BoroCT2020'] = self.gdf['BoroCT2020'].astype(int)
            except (ValueError, TypeError):
                pass
        
        self.gdf_state_plane = self.gdf.to_crs('EPSG:2263')
        self.weights_results = {}
        self.blacklist = None
        
        # Backward compatibility aliases
        self.ct_nyc = self.gdf
        self.ct_state_plane = self.gdf_state_plane

    def set_blacklist(self, blacklist: Dict[int, List[int]], validate: bool = True):
        """
        Set blacklist for custom geometric weights.
        
        Parameters:
        -----------
        blacklist : Dict[int, List[int]]
            Dictionary mapping area indices to lists of indices they shouldn't connect to
        validate : bool
            Whether to validate and show the actual area IDs involved
        """
        if validate and blacklist:
            print("\nValidating blacklist entries:")
            for area_idx, blocked_indices in blacklist.items():
                try:
                    area_id = self.gdf.iloc[area_idx - 1][self.id_column]
                    blocked_area_ids = [self.gdf.iloc[i - 1][self.id_column] for i in blocked_indices]
                    print(f"\nArea index {area_idx} (ID: {area_id}) will not connect to:")
                    for idx, blocked_id in zip(blocked_indices, blocked_area_ids):
                        print(f"  - Index {idx} (ID: {blocked_id})")
                except IndexError as e:
                    raise ValueError(f"Invalid index in blacklist: {e}")
        
        self.blacklist = blacklist
        
    def compute_all_weights(self) -> Dict[str, WeightsAnalysis]:
        """Compute all types of weights matrices."""
        # 1. Queen Contiguity
        queen = libpysal.weights.Queen.from_dataframe(self.gdf)
        W_queen, _ = queen.full()
        n_conn, isolated, avg_conn = analyze_weights(W_queen)
        self.weights_results['queen'] = WeightsAnalysis(
            W_queen, n_conn, isolated, avg_conn,
            "Queen Contiguity from libpysal"
        )

        # 2. Rook Contiguity
        rook = libpysal.weights.Rook.from_dataframe(self.gdf)
        W_rook, _ = rook.full()
        n_conn, isolated, avg_conn = analyze_weights(W_rook)
        self.weights_results['rook'] = WeightsAnalysis(
            W_rook, n_conn, isolated, avg_conn,
            "Rook Contiguity from libpysal"
        )
        
        # 3. Custom Geometric (buffered)
        W_custom = compute_custom_geometric_weights(
                self.gdf_state_plane,
                blacklist=self.blacklist
            )
        n_conn, isolated, avg_conn = analyze_weights(W_custom)
        self.weights_results['custom_geometric'] = WeightsAnalysis(
            W_custom, n_conn, isolated, avg_conn,
            "Custom Geometric with 0.1ft buffer" + 
            (" and blacklist" if self.blacklist else "")
        )
        
        # 4. K-Nearest Neighbors (k=6 as typical default)
        knn = libpysal.weights.KNN.from_dataframe(self.gdf, k=6)
        W_knn, _ = knn.full()
        n_conn, isolated, avg_conn = analyze_weights(W_knn)
        self.weights_results['knn'] = WeightsAnalysis(
            W_knn, n_conn, isolated, avg_conn,
            "6-Nearest Neighbors"
        )
        
        # 5. Distance-based (threshold = median distance to 6 nearest neighbors)
        from sklearn.neighbors import NearestNeighbors
        # Get centroids in state plane for accurate distances
        centroids = np.column_stack([
            self.gdf_state_plane.geometry.centroid.x,
            self.gdf_state_plane.geometry.centroid.y
        ])
        nbrs = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(centroids)
        distances, _ = nbrs.kneighbors(centroids)
        threshold = np.median(distances[:, -1])  # median distance to 6th neighbor
        
        dist = libpysal.weights.DistanceBand.from_array(
            centroids,
            threshold=threshold,
            binary=True
        )
        W_dist, _ = dist.full()
        n_conn, isolated, avg_conn = analyze_weights(W_dist)
        self.weights_results['distance'] = WeightsAnalysis(
            W_dist, n_conn, isolated, avg_conn,
            f"Distance-based (threshold={threshold:.0f}ft)"
        )
        
        return self.weights_results

    def compute_custom_geometric_weights(self, buffer_dist: float = 0.1, debug=False) -> Dict[str, WeightsAnalysis]:
        """Compute custom geometric weights."""
        W = compute_custom_geometric_weights(
            self.gdf_state_plane,
            buffer_dist=buffer_dist,
            blacklist=self.blacklist,
            debug=debug
        )
        n_conn, isolated, avg_conn = analyze_weights(W)
        self.weights_results['custom_geometric'] = WeightsAnalysis(
            W, n_conn, isolated, avg_conn,
            f"Custom Geometric with {buffer_dist}ft buffer" + 
            (" and blacklist" if self.blacklist else "")
        )

        return self.weights_results

    
    def compare_specific_area(self, area_id) -> Dict[str, List]:
        """
        Compare neighbors for a specific area across all methods.
        
        Parameters:
        -----------
        area_id : int or str
            Area ID to analyze (value in id_column)
            
        Returns:
        --------
        Dict[str, List]
            Dictionary of neighbors found by each method
        """
        if not self.weights_results:
            self.compute_all_weights()
            
        area_idx = self.gdf[self.gdf[self.id_column] == area_id].index[0]
        neighbors = {}
        
        for method, analysis in self.weights_results.items():
            W = analysis.weights_matrix
            neighbor_indices = np.where(W[area_idx] == 1)[0]
            neighbor_ids = self.gdf.iloc[neighbor_indices][self.id_column].tolist()
            neighbors[method] = neighbor_ids
            
        return neighbors
    
    # Backward compatibility alias
    def compare_specific_tract(self, tract_id: int) -> Dict[str, List[int]]:
        """Alias for compare_specific_area for backward compatibility."""
        # Try BoroCT2020 first for CT data
        if 'BoroCT2020' in self.gdf.columns:
            area_idx = self.gdf[self.gdf['BoroCT2020'] == tract_id].index[0]
            if not self.weights_results:
                self.compute_all_weights()
            neighbors = {}
            for method, analysis in self.weights_results.items():
                W = analysis.weights_matrix
                neighbor_indices = np.where(W[area_idx] == 1)[0]
                neighbor_ids = self.gdf.iloc[neighbor_indices]['BoroCT2020'].tolist()
                neighbors[method] = neighbor_ids
            return neighbors
        return self.compare_specific_area(tract_id)
    
    def export_adjacency_lists(
        self,
        method: str,
        output_dir: str,
        prefix: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Export adjacency lists for specified method.
        
        Parameters:
        -----------
        method : str
            Which weights method to export
        output_dir : str
            Directory for output files
        prefix : str, optional
            Prefix for output files. Defaults to self.geometry_prefix
            
        Returns:
        --------
        Tuple[str, str]
            Paths to node1 and node2 files
        """
        if method not in self.weights_results:
            raise ValueError(f"Method {method} not found. Run compute_all_weights() first.")
        
        if prefix is None:
            prefix = self.geometry_prefix
            
        W = self.weights_results[method].weights_matrix
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create adjacency lists (1-indexed)
        adj_list = []
        for i in range(W.shape[0]):
            for j in range(i+1, W.shape[1]):  # upper triangle only
                if W[i,j] == 1:
                    adj_list.append([i+1, j+1])
        
        # Write to files
        node1_file = output_dir / f"{prefix}_nyc_adj_list_{method}_node1.txt"
        node2_file = output_dir / f"{prefix}_nyc_adj_list_{method}_node2.txt"
        
        with open(node1_file, "w") as f1, open(node2_file, "w") as f2:
            for pair in adj_list:
                f1.write(f"{pair[0]}\n")
                f2.write(f"{pair[1]}\n")
                
        return str(node1_file), str(node2_file)


# Backward compatibility alias
TractWeightsGenerator = GeometryWeightsGenerator
