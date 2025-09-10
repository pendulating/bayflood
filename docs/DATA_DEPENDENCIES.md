## Data Dependencies (ICAR Artifact Scope)

This document enumerates the datasets used by the ICAR pipeline and their expected locations and formats, so reviewers can understand scope and reproduce results.

### Directory Layout (expected)

```
aggregation/
  geo/
    data/
      ct-nyc-2020.geojson            # NYC census tracts
    data/processed/
      ct_nyc_topology.csv            # Topographic summaries per tract
  flooding/
    data/
      nyc311_flooding_sep29.csv      # 311 flooding complaints
    static/
      current_floodnet_sensors.csv   # FloodNet sensors (current)
      floodnet-flood-sensor-sep-2023.csv
      dep_stormwater_moderate_current/
        data.gdb                     # DEP stormwater polygons (GDB)

data/
  processed/
    flooding_ct_dataset.csv          # Main tract-level dataset
    sep29_positives.csv              # Positive dashcam frames (processed)
    sep29_gt.csv                     # Ground-truth annotations (processed)
  adjacency/
    cg_500/
      ct_nyc_adj_list_custom_geometric_node1.txt
      ct_nyc_adj_list_custom_geometric_node2.txt
```

### Main tract-level dataset: `flooding_ct_dataset.csv`

- Required columns:
  - `GEOID` (string): Census tract identifier
  - `n_total` (int): Total number of images per tract
  - `n_classified_positive` (int): Number of images classified positive for flooding
  - `geometry` (WKT): Tract geometry
- Optional (when `--annotations_have_locations` is set):
  - `n_tp`, `n_fn`, `n_fp`, `n_tn` (int): Manual annotation counts per tract
  - `total_not_annotated` (int), `positives_not_annotated` (int)

This file is the output of the broader data pipeline (dashcam VLM inference + manual annotation integration). That pipeline is out of scope here; reviewers use this processed CSV as input to the ICAR model.

### Spatial adjacency

- Edge lists: `node1.txt` and `node2.txt` represent one-indexed adjacent tract pairs, de-duplicated with `node1[i] < node2[i]`.
- Provided in `data/adjacency/cg_500/` by default. Use `--adj_node1_path`/`--adj_node2_path` to override, or `--adj_npy_path` for a `.npy` adjacency matrix.

### External covariates (when enabled)

Used by `ICAR_prior_annotations_have_locations.stan` via `external_covariates` matrix. Constructed inside `util.read_real_data` from tract-level features, including:
- Topography summaries (`aggregation/geo/data/processed/ct_nyc_topology.csv`)
- DEP stormwater coverage fractions
- FloodNet sensor counts per tract
- 311 complaint counts per tract

### Mapping and analysis inputs

- `aggregation/geo/data/ct-nyc-2020.geojson`: Census tract geometries
- `aggregation/flooding/data/nyc311_flooding_sep29.csv`: 311 complaints
- `aggregation/flooding/static/current_floodnet_sensors.csv`: FloodNet sensors
- `aggregation/flooding/static/dep_stormwater_moderate_current/data.gdb`: DEP stormwater polygons (moderate, current sea levels)
- `data/processed/sep29_positives.csv`, `data/processed/sep29_gt.csv`: Processed image-level positives and ground truth

### Environment configuration

You can override defaults via environment variables or CLI flags:
- `EMPIRICAL_DATA_PATH` → `--empirical_data_path`
- `ADJ_NODE1_PATH`, `ADJ_NODE2_PATH` → `--adj_node1_path`, `--adj_node2_path`
- `ADJ_NPY_PATH` → `--adj_npy_path`
- `EXTERNAL_COVARIATES` (true/false)

See `config.py` and `icar_model.py --help`.
