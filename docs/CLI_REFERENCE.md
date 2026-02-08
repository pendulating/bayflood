## CLI Reference

### pipeline.py (recommended entry point)

```
python pipeline.py \
  --geometry-type {ct,cbg,cb} \
  --prefix STR \
  [--external-covariates] \
  [--no-catch-basins] \
  [--skip-data-generation] \
  [--data-only] \
  [--force-regenerate] \
  [--downsample-frac FLOAT] \
  [--downsample-all-images] \
  [--trim-to-median] \
  [--trim-remove-frac FLOAT] \
  [--compare-to-baselines] \
  [--base-dir PATH]
```

- `--geometry-type`: Census geography level — `ct` (tract), `cbg` (block group), `cb` (block)
- `--prefix`: Prefix for run directory and output files
- `--external-covariates`: Include external covariates (topology, DEP, FloodNet, 311)
- `--no-catch-basins`: Exclude catch basin covariates from external covariates
- `--skip-data-generation`: Skip adjacency/topology/dataset generation; use existing data
- `--data-only`: Only generate data; do not fit the model
- `--force-regenerate`: Force regeneration of all data files even if they exist
- `--downsample-frac`: Fraction of images to keep when downsampling (default: 1.0)
- `--downsample-all-images`: Apply downsampling to all images (annotated + non-annotated)
- `--trim-to-median`: Use trim-to-median moat-fill downsampling variant
- `--trim-remove-frac`: Fraction of images to remove when `--trim-to-median` is set
- `--compare-to-baselines`: Run baseline comparison mode
- `--base-dir`: Override base directory (default: auto-detect from script location)

### icar_model.py

```
python icar_model.py <icar_prior_setting> \
  [--annotations_have_locations] \
  [--simulated_data] \
  [--external_covariates] \
  [--no_catch_basins] \
  [--prefix STR] \
  [--downsample_frac FLOAT] \
  [--downsample_all_images] \
  [--trim_to_median] \
  [--trim_remove_frac FLOAT] \
  [--empirical_data_path PATH] \
  [--adj_node1_path PATH] [--adj_node2_path PATH] \
  [--adj_npy_path PATH] \
  [--geometry_type {ct,cbg,cb}] \
  [--compare_to_baselines]
```

- `icar_prior_setting`: `none | icar | proper | just_model_p_y`
- `--empirical_data_path`: overrides `config.DATASET_PATH` / `EMPIRICAL_DATA_PATH`
- `--adj_*`: specify adjacency inputs (edge lists or `.npy`)
- `--geometry_type`: Census geometry type (default: `ct`)
- `--no_catch_basins`: Exclude catch basins from external covariates
- `--downsample_frac`: Fraction of annotated images to retain (default: 1.0)
- `--downsample_all_images`: Apply downsampling to all images
- `--trim_to_median`: Iterative trim-to-median moat-fill downsampling
- `--trim_remove_frac`: Fraction of images to remove in trim mode
- `--compare_to_baselines`: Run post-processing baseline comparisons

### generate_maps.py (optional)

```
python generate_maps.py <run_id> <estimate_path> <estimate>
```

- `estimate`: `p_y` or `at_least_one_positive_image_by_area`

### aggregation scripts

```bash
# Generate flooding dataset for a geometry type
python aggregation/generate_flooding_dataset.py --geometry-type {ct,cbg,cb} [--base-dir PATH]

# Add covariates to the flooding dataset
python aggregation/add_covariates_to_flooding_dataset.py --geometry-type {ct,cbg,cb} [--base-dir PATH]

# Parameterized aggregation
python aggregation/aggregate_by_geometry.py --geometry-type {ct,cbg,cb}
```
