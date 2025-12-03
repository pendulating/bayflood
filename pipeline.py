#!/usr/bin/env python
"""
BayFlood End-to-End Pipeline

This script runs the complete data processing and model fitting pipeline
for any census geometry type (Census Tracts, Block Groups, or Blocks).

Steps:
1. Generate adjacency network (if not exists)
2. Generate topology statistics (if not exists)
3. Generate flooding dataset (image counts)
4. Add external covariates to flooding dataset
5. Fit ICAR model
6. Generate maps and analysis

Usage:
    # Full pipeline for CBG
    python pipeline.py --geometry-type cbg --prefix cbg_experiment
    
    # With external covariates
    python pipeline.py --geometry-type cbg --prefix cbg_with_cov --external-covariates
    
    # Skip data generation (if already done)
    python pipeline.py --geometry-type cbg --prefix cbg_rerun --skip-data-generation
    
    # Data generation only (no model fitting)
    python pipeline.py --geometry-type cbg --data-only
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))
from geometry_config import GeometryType, get_geometry_paths, get_geometry_config
from logger import setup_logger

logger = setup_logger("bayflood-pipeline")
logger.setLevel("INFO")


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    logger.info(f"Running: {description}")
    logger.info(f"  Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        logger.error(f"Failed: {description}")
        return False
    
    logger.success(f"Completed: {description}")
    return True


def check_data_exists(geometry_type: GeometryType, base_dir: Path) -> dict:
    """Check which data files already exist."""
    paths = get_geometry_paths(geometry_type, str(base_dir))
    
    return {
        'geojson': paths.geojson_path.exists(),
        'adjacency': paths.adjacency_node1_path('custom_geometric').exists(),
        'topology': paths.topology_path.exists(),
        'flooding_dataset': paths.flooding_dataset_path.exists(),
    }


def generate_adjacency(geometry_type: GeometryType, base_dir: Path) -> bool:
    """Generate adjacency network using GeometryWeightsGenerator."""
    logger.info(f"Generating adjacency network for {geometry_type.value}")
    
    # Use Python to generate adjacency
    cmd = [
        sys.executable, '-c', f'''
import sys
sys.path.insert(0, "{base_dir}/notebooks/for_paper/adjacency")
sys.path.insert(0, "{base_dir}")
from tract_weights import GeometryWeightsGenerator
from geometry_config import GeometryType, get_geometry_paths
import os

geometry_type = GeometryType("{geometry_type.value}")
paths = get_geometry_paths(geometry_type, "{base_dir}")

generator = GeometryWeightsGenerator(geometry_type)

# Use appropriate buffer distance
buffer_dist = 500 if geometry_type == GeometryType.CT else (300 if geometry_type == GeometryType.CBG else 100)
generator.compute_custom_geometric_weights(buffer_dist=buffer_dist, debug=True)

os.makedirs(paths.adjacency_dir, exist_ok=True)
generator.export_adjacency_lists("custom_geometric", output_dir=paths.adjacency_dir)
print(f"Adjacency files saved to {{paths.adjacency_dir}}")
'''
    ]
    
    return run_command(cmd, f"Generate adjacency for {geometry_type.value}")


def generate_topology(geometry_type: GeometryType, base_dir: Path) -> bool:
    """Generate topology statistics."""
    cmd = [
        sys.executable,
        str(base_dir / "aggregation" / "geo" / "pp_topology.py"),
        "--geometry-type", geometry_type.value,
        "--base-dir", str(base_dir)
    ]
    return run_command(cmd, f"Generate topology for {geometry_type.value}")


def generate_flooding_dataset(geometry_type: GeometryType, base_dir: Path) -> bool:
    """Generate flooding dataset (image counts and annotations)."""
    cmd = [
        sys.executable,
        str(base_dir / "aggregation" / "generate_flooding_dataset.py"),
        "--geometry-type", geometry_type.value,
        "--base-dir", str(base_dir)
    ]
    return run_command(cmd, f"Generate flooding dataset for {geometry_type.value}")


def add_covariates(geometry_type: GeometryType, base_dir: Path) -> bool:
    """Add external covariates to flooding dataset."""
    cmd = [
        sys.executable,
        str(base_dir / "aggregation" / "add_covariates_to_flooding_dataset.py"),
        "--geometry-type", geometry_type.value,
        "--base-dir", str(base_dir)
    ]
    return run_command(cmd, f"Add covariates for {geometry_type.value}")


def fit_model(
    geometry_type: GeometryType,
    base_dir: Path,
    prefix: str,
    external_covariates: bool = False,
    warmup: int = 6000,
    samples: int = 6000
) -> bool:
    """Fit ICAR model."""
    cmd = [
        sys.executable,
        str(base_dir / "icar_model.py"),
        "icar",
        "--annotations_have_locations",
        "--geometry_type", geometry_type.value,
        "--prefix", prefix,
    ]
    
    if external_covariates:
        cmd.append("--external_covariates")
    
    return run_command(cmd, f"Fit ICAR model for {geometry_type.value}")


def run_pipeline(
    geometry_type: GeometryType,
    base_dir: Path,
    prefix: str,
    external_covariates: bool = False,
    skip_data_generation: bool = False,
    data_only: bool = False,
    force_regenerate: bool = False
):
    """Run the complete pipeline."""
    config = get_geometry_config(geometry_type)
    logger.info(f"="*60)
    logger.info(f"BayFlood Pipeline: {config.display_name}s")
    logger.info(f"="*60)
    
    # Check existing data
    exists = check_data_exists(geometry_type, base_dir)
    logger.info("Existing data:")
    for k, v in exists.items():
        logger.info(f"  {k}: {'✓' if v else '✗'}")
    
    if not skip_data_generation:
        # Step 1: Check/generate adjacency
        if not exists['adjacency'] or force_regenerate:
            if not generate_adjacency(geometry_type, base_dir):
                return False
        else:
            logger.info("Adjacency files exist, skipping...")
        
        # Step 2: Check/generate topology
        if not exists['topology'] or force_regenerate:
            if not generate_topology(geometry_type, base_dir):
                logger.warning("Topology generation failed (DEM may be missing), continuing...")
        else:
            logger.info("Topology file exists, skipping...")
        
        # Step 3: Generate flooding dataset (always regenerate to ensure clean state)
        if not generate_flooding_dataset(geometry_type, base_dir):
            return False
        
        # Step 4: Add covariates
        if not add_covariates(geometry_type, base_dir):
            return False
    else:
        logger.info("Skipping data generation (--skip-data-generation)")
    
    if data_only:
        logger.success("Data generation complete (--data-only mode)")
        return True
    
    # Step 5: Fit model
    if not fit_model(geometry_type, base_dir, prefix, external_covariates):
        return False
    
    logger.success(f"Pipeline complete for {config.display_name}s!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="BayFlood End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--geometry-type',
        type=str,
        required=True,
        choices=['ct', 'cbg', 'cb'],
        help='Census geometry type: ct (tract), cbg (block group), cb (block)'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='experiment',
        help='Prefix for output files and run directory'
    )
    
    parser.add_argument(
        '--external-covariates',
        action='store_true',
        help='Include external covariates in model'
    )
    
    parser.add_argument(
        '--skip-data-generation',
        action='store_true',
        help='Skip data generation steps (use existing data)'
    )
    
    parser.add_argument(
        '--data-only',
        action='store_true',
        help='Only generate data, do not fit model'
    )
    
    parser.add_argument(
        '--force-regenerate',
        action='store_true',
        help='Force regeneration of all data files'
    )
    
    parser.add_argument(
        '--base-dir',
        type=str,
        default=None,
        help='Base directory (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir) if args.base_dir else Path(__file__).parent
    geometry_type = GeometryType(args.geometry_type)
    
    success = run_pipeline(
        geometry_type=geometry_type,
        base_dir=base_dir,
        prefix=args.prefix,
        external_covariates=args.external_covariates,
        skip_data_generation=args.skip_data_generation,
        data_only=args.data_only,
        force_regenerate=args.force_regenerate
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

