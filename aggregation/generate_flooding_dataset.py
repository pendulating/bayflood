#!/usr/bin/env python
"""
Generate flooding dataset aggregated by census geometry.

This script aggregates dashcam image data with predictions and annotations
to any census geometry level (Census Tracts, Block Groups, or Blocks).

This replicates the logic from notebooks/ct_level_dataset.ipynb but
generalized for any census geometry.

The output dataset contains:
- Geography identifiers and geometry
- n_total: total images per area
- n_classified_positive: images classified as flooding (sentiment_1 == 1)
- n_tp, n_fp, n_tn, n_fn: annotation counts by area
- total_not_annotated, positives_not_annotated, negatives_not_annotated

Usage:
    python generate_flooding_dataset.py --geometry-type ct
    python generate_flooding_dataset.py --geometry-type cbg
    python generate_flooding_dataset.py --geometry-type cb
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from geometry_config import GeometryType, get_geometry_config, get_geometry_paths
from logger import setup_logger

logger = setup_logger("flooding-dataset-generator")
logger.setLevel("INFO")

# Constants
WGS = 'EPSG:4326'
PROJ = 'EPSG:2263'


def load_predictions(base_dir: Path) -> pd.DataFrame:
    """
    Load model predictions for images.
    
    Looks for entire_sep29_all.csv which contains sentiment_1 predictions.
    """
    pred_paths = [
        base_dir / "aggregation" / "flooding" / "static" / "entire_sep29_all.csv",
        base_dir / "notebooks" / "entire_sep29_all.csv",
        base_dir / "data" / "processed" / "entire_sep29_all.csv",
        base_dir / "data" / "entire_sep29_all.csv",
    ]
    
    for pred_path in pred_paths:
        if pred_path.exists():
            logger.info(f"Loading predictions from {pred_path}")
            preds = pd.read_csv(pred_path, engine='pyarrow')
            
            # Extract frame_id from image_path (matches notebook logic)
            if 'image_path' in preds.columns:
                preds['frame_id'] = preds['image_path'].apply(
                    lambda x: x.split('/')[-1].split('.')[0].split('_')[-1] if pd.notna(x) else None
                )
            
            logger.info(f"Loaded {len(preds)} prediction records")
            logger.info(f"Positive predictions (sentiment_1==1): {(preds['sentiment_1'] == 1).sum()}")
            return preds
    
    raise FileNotFoundError(f"Prediction file not found. Tried: {pred_paths}")


def load_image_metadata(base_dir: Path) -> pd.DataFrame:
    """
    Load image metadata with coordinates.
    
    Looks for md.csv which contains GPS coordinates for each frame.
    """
    md_paths = [
        Path("/share/ju/urban-fingerprinting/output/default/df/2023-09-29/md.csv"),  # Original location
        base_dir / "data" / "processed" / "md.csv",
        base_dir / "data" / "md.csv",
    ]
    
    for md_path in md_paths:
        if md_path.exists():
            logger.info(f"Loading image metadata from {md_path}")
            md = pd.read_csv(md_path, engine='pyarrow')
            logger.info(f"Loaded {len(md)} image metadata records")
            return md
    
    raise FileNotFoundError(f"Image metadata not found. Tried: {md_paths}")


def load_annotations(base_dir: Path) -> pd.DataFrame:
    """
    Load human annotations for images.
    
    Looks for inspection_set_annotated.csv which contains ground truth labels.
    """
    annot_paths = [
        base_dir / "notebooks" / "inspection_set_annotated.csv",
        base_dir / "data" / "inspection_set_annotated.csv",
        base_dir / "data" / "processed" / "inspection_set_annotated.csv",
    ]
    
    for annot_path in annot_paths:
        if annot_path.exists():
            logger.info(f"Loading annotations from {annot_path}")
            annots = pd.read_csv(annot_path)
            
            # Process annotations exactly as in notebook
            annots['frame_id'] = annots['image'].apply(
                lambda x: x.split('/')[-1].split('.')[0].split('_')[-1] if pd.notna(x) else None
            )
            annots['choice'] = annots['choice'].apply(lambda x: 1 if x == 'Flooded road' else 0)
            annots['pred'] = annots['sentiment_1']
            
            # Calculate tp, fp, tn, fn exactly as in notebook
            annots['tp'] = ((annots['choice'] == 1) & (annots['pred'] == 1)).astype(int)
            annots['fp'] = ((annots['choice'] == 0) & (annots['pred'] == 1)).astype(int)
            annots['tn'] = ((annots['choice'] == 0) & (annots['pred'] == 0)).astype(int)
            annots['fn'] = ((annots['choice'] == 1) & (annots['pred'] == 0)).astype(int)
            
            annots = annots[['frame_id', 'choice', 'pred', 'tp', 'fp', 'tn', 'fn']]
            
            logger.info(f"Loaded {len(annots)} annotation records")
            logger.info(f"Annotations: TP={annots['tp'].sum()}, FP={annots['fp'].sum()}, "
                       f"TN={annots['tn'].sum()}, FN={annots['fn'].sum()}")
            return annots
    
    raise FileNotFoundError(f"Annotation file not found. Tried: {annot_paths}")


def load_census_geometry(geometry_type: GeometryType, base_dir: Path) -> gpd.GeoDataFrame:
    """Load census geometry for spatial joins."""
    paths = get_geometry_paths(geometry_type, str(base_dir))
    config = paths.config
    
    # Try aggregation path first, then main geojson
    geojson_path = paths.aggregation_geojson_path
    if not geojson_path.exists():
        geojson_path = paths.geojson_path
    
    if not geojson_path.exists():
        raise FileNotFoundError(f"Census geometry not found at {geojson_path}")
    
    logger.info(f"Loading {config.display_name} geometry from {geojson_path}")
    gdf = gpd.read_file(str(geojson_path), crs=WGS).to_crs(PROJ)
    
    logger.info(f"Loaded {len(gdf)} {config.display_name}s")
    return gdf


def get_groupby_column(geometry_type: GeometryType, census_gdf: gpd.GeoDataFrame) -> str:
    """
    Determine the best column to use for groupby aggregation.
    
    For CT: prefer BoroCT2020 (matches original notebook)
    For CBG/CB: use GEOID or GEOID20
    """
    config = get_geometry_config(geometry_type)
    
    if geometry_type == GeometryType.CT:
        # Original notebook uses BoroCT2020
        if 'BoroCT2020' in census_gdf.columns:
            return 'BoroCT2020'
    
    # Fall back to standard ID column
    if config.id_column in census_gdf.columns:
        return config.id_column
    
    # Try common alternatives
    for col in ['GEOID', 'GEOID20', 'geoid']:
        if col in census_gdf.columns:
            return col
    
    raise ValueError(f"Could not find ID column in census geometry. Available: {census_gdf.columns.tolist()}")


def generate_flooding_dataset(
    geometry_type: GeometryType,
    base_dir: Path,
    output_dir: Path = None
) -> Path:
    """
    Generate flooding dataset for specified geometry type.
    
    This replicates the exact logic from notebooks/ct_level_dataset.ipynb
    but generalized for any census geometry.
    """
    config = get_geometry_config(geometry_type)
    
    logger.info(f"Generating flooding dataset for {config.display_name}s")
    
    # Step 1: Load census geometry
    census_gdf = load_census_geometry(geometry_type, base_dir)
    groupby_col = get_groupby_column(geometry_type, census_gdf)
    logger.info(f"Using '{groupby_col}' as groupby column")
    
    # Step 2: Load predictions (entire_sep29_all.csv)
    preds = load_predictions(base_dir)
    
    # Step 3: Load image metadata (md.csv)
    md = load_image_metadata(base_dir)
    
    # Step 4: Merge predictions with metadata on frame_id
    logger.info("Merging predictions with metadata...")
    entire_set = md.merge(preds, on='frame_id', how='left')
    logger.info(f"Merged dataset has {len(entire_set)} records")
    
    # Step 5: Create GeoDataFrame from coordinates
    logger.info("Creating point geometries from coordinates...")
    entire_set = entire_set.dropna(subset=['gps_info.longitude', 'gps_info.latitude'])
    entire_set = gpd.GeoDataFrame(
        entire_set,
        geometry=gpd.points_from_xy(
            entire_set['gps_info.longitude'],
            entire_set['gps_info.latitude'],
            crs=WGS
        )
    ).to_crs(PROJ)
    logger.info(f"Created {len(entire_set)} point geometries")
    
    # Step 6: Spatial join with census geometry using sjoin_nearest (matches notebook)
    logger.info("Performing spatial join (sjoin_nearest)...")
    entire_set = gpd.sjoin_nearest(entire_set, census_gdf)
    logger.info(f"After spatial join: {len(entire_set)} records")
    
    # Step 7: Load and merge annotations
    annots = load_annotations(base_dir)
    entire_set = entire_set.merge(annots, on='frame_id', how='left')
    logger.info(f"After annotation merge: {entire_set.isna().sum().sum()} total NaN values")
    
    # Step 8: Aggregate by census geography (matches notebook exactly)
    logger.info(f"Aggregating by {groupby_col}...")
    by_geo = entire_set.groupby(groupby_col).agg({
        'frame_id': 'count',
        'sentiment_1': 'sum',
        'tp': 'sum',
        'fp': 'sum',
        'tn': 'sum',
        'fn': 'sum'
    }).fillna(0).reset_index()
    
    by_geo.columns = [groupby_col, 'n_total', 'n_classified_positive', 'n_tp', 'n_fp', 'n_tn', 'n_fn']
    
    # Step 9: Merge back to census geometry
    result = census_gdf.merge(by_geo, on=groupby_col, how='left').fillna(0)
    
    # Step 10: Calculate derived columns (exact formula from notebook)
    result['total_not_annotated'] = result['n_total'] - (result['n_tp'] + result['n_fp'] + result['n_tn'] + result['n_fn'])
    result['positives_not_annotated'] = result['n_classified_positive'] - (result['n_tp'] + result['n_fp'])
    result['negatives_not_annotated'] = result['n_total'] - result['n_classified_positive'] - result['n_fn']
    
    # Convert count columns to int
    count_cols = ['n_total', 'n_classified_positive', 'n_tp', 'n_fp', 'n_tn', 'n_fn',
                  'total_not_annotated', 'positives_not_annotated', 'negatives_not_annotated']
    for col in count_cols:
        result[col] = result[col].astype(int)
    
    # Log summary statistics
    logger.info("\n=== Summary Statistics ===")
    logger.info(f"Total {config.display_name}s: {len(result)}")
    logger.info(f"n_total sum: {result['n_total'].sum()}")
    logger.info(f"n_classified_positive sum: {result['n_classified_positive'].sum()}")
    logger.info(f"n_tp sum: {result['n_tp'].sum()}")
    logger.info(f"n_fp sum: {result['n_fp'].sum()}")
    logger.info(f"n_tn sum: {result['n_tn'].sum()}")
    logger.info(f"n_fn sum: {result['n_fn'].sum()}")
    logger.info(f"total_not_annotated sum: {result['total_not_annotated'].sum()}")
    logger.info(f"positives_not_annotated sum: {result['positives_not_annotated'].sum()}")
    
    # Step 11: Save output
    if output_dir is None:
        output_dir = base_dir / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"flooding_{geometry_type.value}_dataset.csv"
    result.to_csv(output_path, index=False)
    
    logger.success(f"Generated {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate flooding dataset aggregated by census geometry"
    )
    parser.add_argument(
        '--geometry-type',
        type=str,
        required=True,
        choices=['ct', 'cbg', 'cb'],
        help='Census geometry type: ct (tract), cbg (block group), cb (block)'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=None,
        help='Base directory of bayflood project (default: auto-detect)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: data/processed/)'
    )
    
    args = parser.parse_args()
    
    # Determine base directory
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        base_dir = Path(__file__).parent.parent
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    geometry_type = GeometryType(args.geometry_type)
    
    try:
        output_path = generate_flooding_dataset(geometry_type, base_dir, output_dir)
        print(f"\nDataset generated: {output_path}")
    except Exception as e:
        logger.error(f"Failed to generate dataset: {e}")
        raise


if __name__ == '__main__':
    main()
