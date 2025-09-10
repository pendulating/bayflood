## Components (ICAR Artifact Scope)

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
  - Key methods:
    - `load_data()`: Loads empirical or simulated data, validates inputs, and constructs `observed_data`
    - `fit(CYCLES, WARMUP, SAMPLES, data_already_loaded)`: Builds Stan model per setting; samples and returns `(fit, df)`
    - `plot_results`, `plot_histogram`, `plot_scatter`: Diagnostics and plots
    - `write_estimate`: Writes `estimate_<param>.csv` with CIs
    - `compare_to_baselines`: Train/test split baselines and comparisons
- CLI:
  - `python icar_model.py <icar_prior_setting> [--annotations_have_locations] [--simulated_data] [--external_covariates] [--prefix STR] [--downsample_frac FLOAT] [--empirical_data_path PATH] [--adj_node1_path PATH] [--adj_node2_path PATH] [--adj_npy_path PATH] [--compare_to_baselines]`

### util.py
- Purpose: Data IO, adjacency handling, covariate engineering, simulation, and validation.
- Key functions:
  - `read_real_data(fpath, annotations_have_locations, adj, adj_matrix_storage, use_external_covariates)` → `(observed_data, external_covariates_info)`
  - `validate_observed_data(observed_data, annotations_have_locations, downsample_frac)`
  - `generate_simulated_data(N, images_per_location, total_annotated_classified_negative, total_annotated_classified_positive, icar_prior_setting, annotations_have_locations)`

### analysis_df.py
- Purpose: Merge ICAR run estimates with tract geometries, ACS features, topology summaries, FloodNet sensors, DEP stormwater coverage, and 311 counts to produce tract-level CSVs.
- Main function: `generate_nyc_analysis_df(run_dir, custom_prefix, use_smoothing, base_dir='.', logger=None)` → `pd.DataFrame`
- Inputs: expects estimate CSVs in `run_dir`, and data per `docs/DATA_DEPENDENCIES.md`.

### generate_maps.py (optional)
- Purpose: Visualize tract-level estimates with overlays of positives, ground truth, FloodNet sensors, 311, and DEP polygons.
- Main function: `generate_maps(run_id, estimate_path, estimate='p_y' | 'at_least_one_positive_image_by_area')`

### logger.py
- Purpose: Colored logging with custom `SUCCESS` level; `setup_logger(name)` standardizes console logs.

### refresh_cache.py
- Purpose: Clear local Stan cache directory for a clean rebuild; `refresh_cache(base_dir=None)`.

### config.py
- Purpose: Centralize defaults and environment overrides for paths and sampling params.
- Exposed:
  - `DATASET_PATH`, `ADJ_NODE1_PATH`, `ADJ_NODE2_PATH`, `ADJ_NPY_PATH`
  - `EXTERNAL_COVARIATES`, `DEFAULT_WARMUP`, `DEFAULT_SAMPLES`
