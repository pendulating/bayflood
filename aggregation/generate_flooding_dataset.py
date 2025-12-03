#!/usr/bin/env python
"""
Generate flooding dataset aggregated by census geometry.

This script aggregates dashcam image data with predictions and annotations
to any census geometry level (Census Tracts, Block Groups, or Blocks).

The output dataset contains:
- Geography identifiers and geometry
- n_total: total images per area
- n_classified_positive: images classified as flooding
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
from datetime import datetime

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


def load_image_metadata(base_dir: Path) -> pd.DataFrame:
    """Load image metadata with coordinates."""
    md_path = base_dir / "data" / "processed" / "md.csv"
    if not md_path.exists():
        md_path = base_dir / "data" / "md.csv"
    
    if not md_path.exists():
        raise FileNotFoundError(f"Image metadata not found at {md_path}")
    
    logger.info(f"Loading image metadata from {md_path}")
    md = pd.read_csv(md_path, engine='pyarrow')
    
    # Ensure we have coordinate columns
    required_cols = ['frame_id', 'gps_info.longitude', 'gps_info.latitude']
    missing = [c for c in required_cols if c not in md.columns]
    if missing:
        raise ValueError(f"Missing required columns in metadata: {missing}")
    
    logger.info(f"Loaded {len(md)} image records")
    return md


def load_predictions(base_dir: Path) -> pd.DataFrame:
    """Load model predictions for images."""
    # Try different possible prediction file locations
    pred_paths = [
        base_dir / "data" / "processed" / "sep29_predictions.csv",
        base_dir / "data" / "processed" / "entire_sep29_all.csv",
    ]
    
    for pred_path in pred_paths:
        if pred_path.exists():
            logger.info(f"Loading predictions from {pred_path}")
            preds = pd.read_csv(pred_path, engine='pyarrow')
            
            # Normalize frame_id column
            if 'image_path' in preds.columns and 'frame_id' not in preds.columns:
                preds['frame_id'] = preds['image_path'].apply(
                    lambda x: x.split('/')[-1].split('.')[0] if pd.notna(x) else None
                )
            
            # Normalize sentiment column
            if 'sentiment_1' in preds.columns and 'pred' not in preds.columns:
                preds['pred'] = preds['sentiment_1']
            
            logger.info(f"Loaded {len(preds)} prediction records")
            return preds
    
    logger.warning("No prediction file found, will use metadata only")
    return None


def load_annotations(base_dir: Path) -> pd.DataFrame:
    """Load human annotations for images."""
    annot_paths = [
        base_dir / "data" / "processed" / "bayflood_inspection_set.csv",
        base_dir / "data" / "inspection_set_annotated.csv",
    ]
    
    for annot_path in annot_paths:
        if annot_path.exists():
            logger.info(f"Loading annotations from {annot_path}")
            annots = pd.read_csv(annot_path)
            
            # Normalize frame_id column
            if 'image' in annots.columns and 'frame_id' not in annots.columns:
                annots['frame_id'] = annots['image'].apply(
                    lambda x: 'nlbx_' + x.split('/')[-1].split('.')[0].split('_')[-1] if pd.notna(x) else None
                )
            
            logger.info(f"Loaded {len(annots)} annotation records")
            return annots
    
    logger.warning("No annotation file found")
    return None


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
    gdf = gpd.read_file(str(geojson_path))
    gdf = gdf.to_crs(PROJ)
    
    logger.info(f"Loaded {len(gdf)} {config.display_name}s")
    return gdf


def aggregate_to_geometry(
    images_gdf: gpd.GeoDataFrame,
    census_gdf: gpd.GeoDataFrame,
    id_column: str,
    has_predictions: bool = True,
    has_annotations: bool = True
) -> pd.DataFrame:
    """
    Aggregate image data to census geometry level.
    
    Parameters
    ----------
    images_gdf : GeoDataFrame
        Point GeoDataFrame with image locations and predictions/annotations
    census_gdf : GeoDataFrame
        Polygon GeoDataFrame with census geometries
    id_column : str
        Column name for geography ID
    has_predictions : bool
        Whether prediction columns are available
    has_annotations : bool
        Whether annotation columns are available
        
    Returns
    -------
    pd.DataFrame
        Aggregated counts per geography
    """
    logger.info("Performing spatial join...")
    
    # Spatial join: find which census area each image falls into
    joined = gpd.sjoin(images_gdf, census_gdf[[id_column, 'geometry']], how='inner', predicate='within')
    logger.info(f"Joined {len(joined)} images to census geometries")
    
    # Aggregate counts
    agg_dict = {
        'frame_id': 'count',  # n_total
    }
    
    if has_predictions and 'pred' in joined.columns:
        agg_dict['pred'] = 'sum'  # n_classified_positive
    
    if has_annotations:
        for col in ['tp', 'fp', 'tn', 'fn']:
            if col in joined.columns:
                agg_dict[col] = 'sum'
    
    logger.info("Aggregating counts by geography...")
    aggregated = joined.groupby(id_column).agg(agg_dict).reset_index()
    
    # Rename columns
    rename_map = {
        'frame_id': 'n_total',
        'pred': 'n_classified_positive',
        'tp': 'n_tp',
        'fp': 'n_fp', 
        'tn': 'n_tn',
        'fn': 'n_fn',
    }
    aggregated = aggregated.rename(columns={k: v for k, v in rename_map.items() if k in aggregated.columns})
    
    # Calculate derived columns
    if 'n_tp' in aggregated.columns:
        aggregated['total_annotated'] = (
            aggregated.get('n_tp', 0) + 
            aggregated.get('n_fp', 0) + 
            aggregated.get('n_tn', 0) + 
            aggregated.get('n_fn', 0)
        )
        aggregated['total_not_annotated'] = aggregated['n_total'] - aggregated['total_annotated']
        
        # Positives not annotated
        if 'n_classified_positive' in aggregated.columns:
            annotated_positives = aggregated.get('n_tp', 0) + aggregated.get('n_fp', 0)
            aggregated['positives_not_annotated'] = aggregated['n_classified_positive'] - annotated_positives
            aggregated['negatives_not_annotated'] = aggregated['total_not_annotated'] - aggregated['positives_not_annotated']
    
    logger.info(f"Aggregated to {len(aggregated)} unique geographies")
    return aggregated


def generate_flooding_dataset(
    geometry_type: GeometryType,
    base_dir: Path,
    output_dir: Path = None
) -> Path:
    """
    Generate flooding dataset for specified geometry type.
    
    Parameters
    ----------
    geometry_type : GeometryType
        Census geometry level (CT, CBG, CB)
    base_dir : Path
        Base directory of the bayflood project
    output_dir : Path, optional
        Output directory (defaults to data/processed/)
        
    Returns
    -------
    Path
        Path to generated dataset
    """
    config = get_geometry_config(geometry_type)
    id_column = config.id_column
    
    logger.info(f"Generating flooding dataset for {config.display_name}s")
    
    # Load data
    md = load_image_metadata(base_dir)
    preds = load_predictions(base_dir)
    annots = load_annotations(base_dir)
    census_gdf = load_census_geometry(geometry_type, base_dir)
    
    # Merge metadata with predictions and annotations
    if preds is not None:
        md = md.merge(preds[['frame_id', 'pred']], on='frame_id', how='left')
        md['pred'] = md['pred'].fillna(0).astype(int)
    
    if annots is not None:
        annot_cols = ['frame_id'] + [c for c in ['gt', 'pred', 'tp', 'fp', 'tn', 'fn'] if c in annots.columns]
        md = md.merge(annots[annot_cols], on='frame_id', how='left', suffixes=('', '_annot'))
        for col in ['tp', 'fp', 'tn', 'fn']:
            if col in md.columns:
                md[col] = md[col].fillna(0).astype(int)
    
    # Create GeoDataFrame from images
    logger.info("Creating image point geometries...")
    md = md.dropna(subset=['gps_info.longitude', 'gps_info.latitude'])
    images_gdf = gpd.GeoDataFrame(
        md,
        geometry=gpd.points_from_xy(md['gps_info.longitude'], md['gps_info.latitude']),
        crs=WGS
    ).to_crs(PROJ)
    
    logger.info(f"Created {len(images_gdf)} image points with valid coordinates")
    
    # Aggregate to geometry
    aggregated = aggregate_to_geometry(
        images_gdf,
        census_gdf,
        id_column,
        has_predictions=(preds is not None),
        has_annotations=(annots is not None)
    )
    
    # Merge with full census geometry (to include areas with 0 images)
    result = census_gdf.merge(aggregated, on=id_column, how='left')
    
    # Fill NaN counts with 0
    count_cols = ['n_total', 'n_classified_positive', 'n_tp', 'n_fp', 'n_tn', 'n_fn',
                  'total_not_annotated', 'positives_not_annotated', 'negatives_not_annotated']
    for col in count_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0).astype(int)
    
    # Output
    if output_dir is None:
        output_dir = base_dir / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"flooding_{geometry_type.value}_dataset.csv"
    result.to_csv(output_path, index=False)
    
    logger.success(f"Generated {output_path}")
    logger.info(f"Total {config.display_name}s: {len(result)}")
    logger.info(f"Total images: {result['n_total'].sum()}")
    if 'n_classified_positive' in result.columns:
        logger.info(f"Total classified positive: {result['n_classified_positive'].sum()}")
    
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

