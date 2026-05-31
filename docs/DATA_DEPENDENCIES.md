## Data Dependencies

This document enumerates the datasets used by the BayFlood pipeline and their expected locations and formats.

### Data provenance & availability

Each input falls into one of three categories, summarizing the project's
data-availability terms:

- **Shipped** — included in this repository; enough to reproduce the paper's
  reported numbers from the precomputed `FINAL_*` runs.
- **Fetched** — downloaded from a public source via a provided script.
- **Embargoed/External** — not redistributable here; obtained under a research
  agreement or too large to ship. The analyses that need these are documented as
  requiring external access.

| Dataset | Path (root-relative) | Category | Source / notes |
|---|---|---|---|
| Aggregated tract flooding counts | `data/processed/flooding_ct_dataset.csv` | Shipped | Model input (counts per tract); derived from VLM scan |
| Final model runs (cov / no-cov) | `runs/.../FINAL_20260206-*/` | Shipped | Precomputed ICAR posterior + analysis DataFrames |
| Tract adjacency lists | `data/adjacency/cg_500/`, `cbg_cg_300/` | Shipped | ICAR neighborhood structure |
| ACS demographics (2023 5-yr) | `aggregation/demo/data/acs2023_*.json` | Shipped | U.S. Census Bureau ACS |
| 311 flooding + clogged-CB | `aggregation/flooding/data/nyc311_*.csv` | Shipped | NYC OpenData 311 |
| FloodNet sensors | `aggregation/flooding/static/current_floodnet_sensors.csv` | Shipped | FloodNet NYC |
| DEP stormwater map | `aggregation/flooding/static/dep_stormwater_moderate_current/` | Shipped | NYC DEP |
| Catch basins | `aggregation/flooding/static/catch_basins_nyc.geojson` | Shipped | NYC OpenData |
| Census Tract boundary | `aggregation/geo/data/ct-nyc-2020.geojson` | Shipped | Census TIGER/Line (default geometry) |
| Inspection set + IRR annotations | `data/processed/inspection_set.csv`, `data/revisions/irr/bayflood_annotator{1,2,3}.csv` | Shipped | Human annotations + VLM responses |
| Prompt-baseline annotations | `data/revisions/prompt_baseline_annotations/*.csv` | Shipped | Alternate-prompt VLM outputs |
| Census Block Group / Block boundaries | `aggregation/geo/data/{cbg,cb}-nyc-2020.geojson` | Fetched | `aggregation/geo/pull-data.sh` (Census TIGER/Line) |
| Raw dashcam imagery | `/share/ju/nexar_data/...` | Embargoed/External | Nexar, under research-evaluation agreement; **not redistributable** |
| Full VLM image scan | `data/processed/{md.csv, entire_sep29_all.csv}`, `notebooks/cambrian/*` | Embargoed/External | ~0.4–0.9 GB per-image VLM Q&A; derived from embargoed imagery |
| DEM LiDAR raster | `data/DEM_LiDAR_1ft_2010_Improved_NYC_int.tif` | Embargoed/External | 3.2 GB; NYC LiDAR DEM (download separately) |
| VLM baseline annotations | `notebooks/for_paper/vlm_baselines/{cambrian-8b,clip-vitg,januspro_onefoot,supervised}_annotations.csv` | Shipped | Per-image classifications for the 4 baseline models in `tab:vlm-baselines` (Supervised, CLIP, Janus-Pro, Cambrian-8B); reproduced exactly by `f_vlm_baselines_ttest.ipynb` |

> Large embargoed/external files are gitignored. The model-output analyses
> (`notebooks/for_final/`, most of `for_revisions/`) run entirely from the
> **Shipped** category; only image-level analyses need the external imagery/scans.

### Directory Layout

```
aggregation/
  geo/
    pull-data.sh                       # Script to download GeoJSON boundary files
    data/
      ct-nyc-2020.geojson             # NYC Census Tracts (downloaded via script)
      cbg-nyc-2020.geojson            # NYC Census Block Groups (downloaded via script)
      cb-nyc-2020.geojson             # NYC Census Blocks (downloaded via script)
      ct-nyc-wi-2020.geojson          # Tracts including water (downloaded via script)
    data/processed/
      ct_nyc_topology.csv             # Topographic summaries per tract
      cbg_nyc_topology.csv            # Topographic summaries per block group
  flooding/
    data/
      nyc311_flooding_sep29.csv       # 311 flooding complaints
    static/
      current_floodnet_sensors.csv    # FloodNet sensors (current)
      floodnet-flood-sensor-sep-2023.csv
      dep_stormwater_moderate_current/
        data.gdb                      # DEP stormwater polygons (GDB)
  demo/
    data/
      acs2023_dp05.json               # ACS: demographics
      acs2023_s2801.json              # ACS: internet access
      acs2023_s1901.json              # ACS: income
      acs2023_s1501.json              # ACS: education
      acs2023_s1602.json              # ACS: language

data/
  processed/
    flooding_ct_dataset.csv           # Processed flooding dataset (Census Tracts)
    sep29_positives.csv               # Positive dashcam frames (processed)
    sep29_gt.csv                      # Ground-truth annotations (processed)
    inspection_set.csv                # Human inspection/annotation set
  adjacency/
    cg_500/                           # Census Tract adjacency (500ft buffer)
      ct_nyc_adj_list_custom_geometric_node1.txt
      ct_nyc_adj_list_custom_geometric_node2.txt
    cbg_cg_300/                       # Block Group adjacency (300ft buffer)
      cbg_nyc_adj_list_custom_geometric_node1.txt
      cbg_nyc_adj_list_custom_geometric_node2.txt
  revisions/                          # Data for revision analyses
    irr/
      bayflood_annotator1.csv         # IRR: annotator 1 (original human gt labels)
      bayflood_annotator2.csv         # IRR: annotator 2 (revision)
      bayflood_annotator3.csv         # IRR: annotator 3 (revision)
      inspection_set_IRR.csv          # IRR sample frame list
    nearby_floodnet/
      floodnet_depth_*.csv            # FloodNet sensor depth data
    prompt_baseline_annotations/
      *.csv                           # VLM prompt baseline annotation results
    04_image_sensor_proximity.json    # Sensor proximity analysis data

runs/
  icar_icar/simulated_False/ahl_True/
    covariates_True/FINAL_20260206-1100/    # Final run (with covariates)
    covariates_False/FINAL_20260206-1205/   # Final run (without covariates)
```

### Downloading GeoJSON boundary files

The **Census Tract** boundary used by the default pipeline
(`aggregation/geo/data/ct-nyc-2020.geojson`) and the catch-basin layer
(`aggregation/flooding/static/catch_basins_nyc.geojson`) **are tracked in the
repository**. The larger **Census Block Group (CBG)** and **Census Block (CB)**
boundaries are not tracked — download them with the provided script:

```bash
cd aggregation/geo && bash pull-data.sh && cd ../..
```

This fetches the CBG and CB GeoJSON files (and refreshes the CT file) from the
U.S. Census Bureau TIGER/Line program for NYC (2020). You only need CBG/CB if you
run the pipeline at those geometry levels (`--geometry-type cbg|cb`).

### Main flooding dataset: `flooding_ct_dataset.csv`

- Default path: `data/processed/flooding_ct_dataset.csv`
- For other geometries: `data/processed/flooding_{prefix}_dataset.csv` (e.g., `flooding_cbg_dataset.csv`)
- Required columns (subset used by the ICAR pipeline):
  - `GEOID` (string): Census geometry identifier
  - `n_total` (int): Total number of images per geometry unit
  - `n_classified_positive` (int): Number of images classified positive for flooding
  - (optional) When using annotation locations: `n_tp`, `n_fn`, `n_fp`, `n_tn`, `total_not_annotated`, `positives_not_annotated`

This file is the output of the data processing pipeline (dashcam VLM inference + manual annotation integration). The processing scripts are in `aggregation/`.

### Spatial adjacency

- Edge lists: `node1.txt` and `node2.txt` represent one-indexed adjacent geometry pairs, de-duplicated with `node1[i] < node2[i]`.
- Defaults per geometry:
  - Census Tracts: `data/adjacency/cg_500/` (500 ft buffer)
  - Block Groups: `data/adjacency/cbg_cg_300/` (300 ft buffer)
  - Census Blocks: `data/adjacency/cb_cg_100/` (100 ft buffer)
- Use `--adj_node1_path`/`--adj_node2_path` to override, or `--adj_npy_path` for a `.npy` adjacency matrix.
- Adjacency networks are generated by `GeometryWeightsGenerator` in `notebooks/for_paper/adjacency/tract_weights.py`.

### External covariates (when enabled)

Used by `ICAR_prior_annotations_have_locations.stan` via the `external_covariates` matrix. Constructed inside `util.read_real_data` from geometry-level features, including:
- Topography summaries (`aggregation/geo/data/processed/{prefix}_nyc_topology.csv`)
- DEP stormwater coverage fractions
- FloodNet sensor counts per geometry
- 311 complaint counts per geometry
 
#### ACS demographic features (used in analysis)
- Location: `aggregation/demo/data/`
- Files (2023): `acs2023_dp05.json`, `acs2023_s2801.json`, `acs2023_s1901.json`, `acs2023_s1501.json`, `acs2023_s1602.json`

### Multi-geometry support

The pipeline supports three census geography levels:

| Geometry | Prefix | ID Column | Default Buffer | GeoJSON |
|----------|--------|-----------|----------------|---------|
| Census Tract | `ct` | `GEOID` | 500 ft | `ct-nyc-2020.geojson` |
| Block Group | `cbg` | `GEOID` | 300 ft | `cbg-nyc-2020.geojson` |
| Census Block | `cb` | `GEOID20` | 100 ft | `cb-nyc-2020.geojson` |

All path resolution is handled by `geometry_config.py`, which provides a `GeometryPaths` factory class.

### Environment configuration

You can override defaults via environment variables or CLI flags:
- `EMPIRICAL_DATA_PATH` → `--empirical_data_path`
- `ADJ_NODE1_PATH`, `ADJ_NODE2_PATH` → `--adj_node1_path`, `--adj_node2_path`
- `ADJ_NPY_PATH` → `--adj_npy_path`
- `EXTERNAL_COVARIATES` (true/false)
- `BAYFLOOD_GEOMETRY_TYPE` → `--geometry_type` (default: `ct`)

See `config.py` and `icar_model.py --help`.
