## Components

### pipeline.py (NEW)
- Purpose: End-to-end pipeline orchestrating data generation, model fitting, and analysis for any census geometry type.
- Steps:
  1. Generate adjacency network (via `GeometryWeightsGenerator`)
  2. Generate topology statistics (via `pp_topology.py`)
  3. Generate flooding dataset (via `generate_flooding_dataset.py`)
  4. Add external covariates (via `add_covariates_to_flooding_dataset.py`)
  5. Fit ICAR model (via `icar_model.py`)
  6. Copy context dataframe to run directory
- CLI: `python pipeline.py --geometry-type {ct,cbg,cb} --prefix STR [options]`
- Supports: `--external-covariates`, `--skip-data-generation`, `--data-only`, `--force-regenerate`, `--downsample-frac`, `--downsample-all-images`, `--trim-to-median`, `--compare-to-baselines`

### geometry_config.py (NEW)
- Purpose: Centralized configuration for multi-geometry support (Census Tracts, Block Groups, Blocks).
- Key types:
  - `GeometryType` enum: `CT`, `CBG`, `CB`
  - `GeometryConfig` dataclass: display name, ID column, file prefix, default adjacency buffer
  - `GeometryPaths` class: path factory for geometry-specific file paths (GeoJSON, adjacency, datasets, topology, runs)
- Factory: `get_geometry_paths(geometry_type, base_dir)` → `GeometryPaths`
- Default geometry type controlled by `BAYFLOOD_GEOMETRY_TYPE` environment variable (default: `ct`).

### icar_model.py
- Purpose: Train ICAR/CAR-based Bayesian models in Stan via the Python `pystan` backend; manage runs and outputs.
- Main class: `ICAR_MODEL`
  - Key init args:
    - `PREFIX`: run prefix used in `runs/<...>`
    - `ICAR_PRIOR_SETTING`: one of `"none" | "icar" | "proper" | "just_model_p_y"`
    - `ANNOTATIONS_HAVE_LOCATIONS`: bool; enables annotation-location model and external covariates pathway
    - `EXTERNAL_COVARIATES`: bool; when true, builds `external_covariates` matrix inside `util.read_real_data`
    - `SIMULATED_DATA`: bool; use simulated data generation in `util.generate_simulated_data`
    - `ESTIMATE_PARAMS`: subset of `["p_y", "at_least_one_positive_image_by_area", "at_least_one_positive_image_by_area_if_you_have_100_images"]`
    - `EMPIRICAL_DATA_PATH`: path to processed dataset CSV
    - `adj`: adjacency input paths (edge lists or `.npy`)
    - `adj_matrix_storage`: True if `.npy` adjacency path provided
    - `downsample_frac`: float, downsampling of annotated images
    - `GEOMETRY_TYPE`: geometry type string (`ct`, `cbg`, `cb`)
  - Key methods:
    - `load_data()`: Loads empirical or simulated data, validates inputs, and constructs `observed_data`
    - `fit(CYCLES, WARMUP, SAMPLES, data_already_loaded)`: Builds Stan model per setting; samples and returns `(fit, df)`
    - `plot_results`, `plot_histogram`, `plot_scatter`: Diagnostics and plots
    - `write_estimate`: Writes `estimate_<param>.csv` with CIs
    - `compare_to_baselines`: Train/test split baselines and comparisons
- CLI:
  - `python icar_model.py <icar_prior_setting> [--annotations_have_locations] [--simulated_data] [--external_covariates] [--no_catch_basins] [--prefix STR] [--downsample_frac FLOAT] [--downsample_all_images] [--trim_to_median] [--trim_remove_frac FLOAT] [--empirical_data_path PATH] [--adj_node1_path PATH] [--adj_node2_path PATH] [--adj_npy_path PATH] [--geometry_type {ct,cbg,cb}] [--compare_to_baselines]`

### util.py
- Purpose: Data IO, adjacency handling, covariate engineering, simulation, and validation.
- Key functions:
  - `read_real_data(fpath, annotations_have_locations, adj, adj_matrix_storage, use_external_covariates)` → `(observed_data, external_covariates_info)`
  - `validate_observed_data(observed_data, annotations_have_locations, downsample_frac)`
  - `generate_simulated_data(N, images_per_location, total_annotated_classified_negative, total_annotated_classified_positive, icar_prior_setting, annotations_have_locations)`

### analysis_df.py
- Purpose: Merge ICAR run estimates with geometry boundaries, ACS features, topology summaries, FloodNet sensors, DEP stormwater coverage, and 311 counts to produce analysis CSVs.
- Main function: `generate_nyc_analysis_df(run_dir, custom_prefix, use_smoothing, base_dir='.', logger=None)` → `pd.DataFrame`
- Inputs: expects estimate CSVs in `run_dir`, and data per `docs/DATA_DEPENDENCIES.md`.

### generate_maps.py (optional)
- Purpose: Visualize geometry-level estimates with overlays of positives, ground truth, FloodNet sensors, 311, and DEP polygons.
- Main function: `generate_maps(run_id, estimate_path, estimate='p_y' | 'at_least_one_positive_image_by_area')`

### aggregation/generate_flooding_dataset.py
- Purpose: Generate the flooding dataset (image counts and annotations per geometry unit) from raw inference outputs.
- Parameterized by geometry type via `--geometry-type`.

### aggregation/add_covariates_to_flooding_dataset.py
- Purpose: Add external covariates (topology, DEP stormwater, FloodNet, 311) to the flooding dataset.
- Parameterized by geometry type via `--geometry-type`.

### aggregation/aggregate_by_geometry.py (NEW)
- Purpose: Parameterized aggregation of flooding data and covariates to different census geography levels.
- CLI: `python aggregate_by_geometry.py --geometry-type {ct,cbg,cb}`

### notebooks/for_paper/adjacency/tract_weights.py
- Purpose: Generate and analyze spatial weights for census geographies.
- Key class: `GeometryWeightsGenerator` — supports custom geometric buffer, queen/rook contiguity, and distance-band adjacency methods. Used by `pipeline.py` to generate adjacency networks.

### logger.py
- Purpose: Colored logging with custom `SUCCESS` level; `setup_logger(name)` standardizes console logs.

### refresh_cache.py
- Purpose: Clear local Stan cache directory for a clean rebuild; `refresh_cache(base_dir=None)`.

### config.py
- Purpose: Centralize defaults and environment overrides for paths and sampling params.
- Exposed:
  - `DATASET_PATH`, `ADJ_NODE1_PATH`, `ADJ_NODE2_PATH`, `ADJ_NPY_PATH`
  - `EXTERNAL_COVARIATES`, `DEFAULT_WARMUP`, `DEFAULT_SAMPLES`
