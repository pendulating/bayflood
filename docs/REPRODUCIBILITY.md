## Reproducibility Guide

This guide explains how to reproduce the main results using precomputed run outputs, how to regenerate analyses, and how to run the full pipeline for different census geometries.

### 1) Precomputed runs

Two FINAL model runs are included in the repository:

- **With covariates**:
  `runs/icar_icar/simulated_False/ahl_True/covariates_True/FINAL_20260206-1100`
- **Without covariates**:
  `runs/icar_icar/simulated_False/ahl_True/covariates_False/FINAL_20260206-1205`

Each run directory contains:
- `analysis_df_FINAL_02062026.csv` — Full tract-level analysis DataFrame
- `analysis_df_describe_FINAL_02062026.csv` — Descriptive statistics
- `metadata.json` — Run configuration and parameters
- `summary.txt` — Stan sampling summary

### 2) Generate tract-level analysis CSVs from a precomputed run

```python
from analysis_df import generate_nyc_analysis_df

# With covariates
df_cov = generate_nyc_analysis_df(
    run_dir='runs/icar_icar/simulated_False/ahl_True/covariates_True/FINAL_20260206-1100',
    custom_prefix='with_covariates',
    use_smoothing=True
)

# Without covariates
df_nocov = generate_nyc_analysis_df(
    run_dir='runs/icar_icar/simulated_False/ahl_True/covariates_False/FINAL_20260206-1205',
    custom_prefix='without_covariates',
    use_smoothing=True
)
```

Outputs are written back to each `run_dir` (e.g., `analysis_df_*.csv` and `analysis_df_describe_*.csv`).

### 3) (Optional) Generate maps from a run

`generate_maps.py` is optional and requires geopandas + matplotlib.

```python
from generate_maps import generate_maps

generate_maps(
    run_id='icar_icar/simulated_False/ahl_True/covariates_True/FINAL_20260206-1100',
    estimate_path='runs/icar_icar/simulated_False/ahl_True/covariates_True/FINAL_20260206-1100/estimate_p_y.csv',
    estimate='p_y'
)
```

### 4) Regenerating runs from scratch

#### Using the end-to-end pipeline (recommended)

The `pipeline.py` script orchestrates the entire workflow—data generation, model fitting, and analysis—for any census geometry:

```bash
# Full pipeline for Census Tracts with covariates
python pipeline.py --geometry-type ct --prefix REPRO --external-covariates

# Full pipeline for Census Block Groups
python pipeline.py --geometry-type cbg --prefix REPRO_CBG --external-covariates

# Data generation only (no model fitting)
python pipeline.py --geometry-type ct --data-only
```

The pipeline steps are:
1. Generate adjacency network (if not present)
2. Generate topology statistics (if DEM raster available)
3. Generate flooding dataset (image counts per geometry)
4. Add external covariates
5. Fit ICAR model
6. Copy context dataframe to run directory

#### Using individual commands

To re-run training directly:

```bash
python icar_model.py icar \
  --annotations_have_locations \
  --external_covariates \
  --geometry_type ct \
  --prefix REPRO_REDO \
  --empirical_data_path data/processed/flooding_ct_dataset.csv \
  --adj_node1_path data/adjacency/cg_500/ct_nyc_adj_list_custom_geometric_node1.txt \
  --adj_node2_path data/adjacency/cg_500/ct_nyc_adj_list_custom_geometric_node2.txt
```

Then run the analysis step on the new run directory as in step (2).

### 5) Downloading geospatial boundary files

Census boundary GeoJSON files are not stored in the repository. Download them with:

```bash
cd aggregation/geo && bash pull-data.sh && cd ../..
```

This fetches Census Tract, Block Group, and Block boundaries from the U.S. Census Bureau.

### 5b) Post-processing baseline comparison (`tab:baselines`)

The baseline-comparison table (`tab:baselines`: BayFlood vs. graph-Laplacian,
OLS, RandomForest, raw-classification baselines over 20 train/test splits) is
reproducible from shipped data — the 20 baseline runs are committed:

```
runs/icar_icar/simulated_False/ahl_True/covariates_True/FINAL_COMMS_BASELINES_*/
    performance_on_baselines.csv
```

Aggregate them (mean ± 95% CI over the 20 splits) with
`notebooks/for_paper/f_postprocessing_baselines.ipynb`. The BayFlood row reports
the Pearson r of the continuous estimate (`bayesian_model_p_y`, 0.61 ± 0.03) and
the detection AUCs of the binary indicator (`bayesian_model_at_least_one_positive_by_area`,
0.87 / 0.79).

To regenerate the runs from scratch (heavy; needs the Stan/pystan env), each run is:

```bash
python pipeline.py --geometry-type ct --external-covariates \
    --skip-data-generation --compare-to-baselines --prefix FINAL_COMMS_BASELINES
```

Submit 20 in sequence on SLURM with `bash jobs/baselines_sequence.sh 20` (each
uses a different random train/test split).

### 6) Revision-specific notebooks

Revision notebooks live in `notebooks/for_revisions/`. Some depend on run outputs that may need to be regenerated (see the note cells at the top of those notebooks). Data used by revision analyses is in `data/revisions/`.

### Two environments / two levels of reproduction

Reproduction comes in two scopes, with different requirements:

1. **Reproduce the paper's reported numbers from the precomputed runs** (fast, no
   GPU, no Stan). The shipped `FINAL_*` runs + the analysis notebooks
   (`notebooks/for_final/`, `notebooks/for_revisions/`) regenerate the figures,
   tables, and statistics. This is the environment pinned in
   **`requirements-lock.txt`** (Python 3.12, pandas/numpy/scikit-learn/statsmodels/
   geopandas). A per-number audit is in `docs/NUMBER_VERIFICATION.md`.
2. **Re-fit the ICAR model from scratch** (the `FINAL_*` runs). This additionally
   requires the **`pystan`** backend (`pip install -r requirements.txt`) and the
   aggregated `flooding_*_dataset.csv`. It does **not** require the raw dashcam
   imagery (that feeds only the upstream VLM inference, which is embargoed).

### Notes

- The analysis/verification environment is **Python 3.12** (see
  `requirements-lock.txt`). The full model-fitting pipeline targets **Python 3.10**
  with `pystan>=3.7` (see `requirements.txt`); install that separately if re-fitting.
- See `docs/DATA_DEPENDENCIES.md` for required datasets and expected paths.
- Model fitting requires at least 8 CPU cores and 64 GB RAM; typical runtime is ~20 minutes per run at default settings.
- For exact citation and licensing, see `CITATION.cff` and `LICENSE.md` at the repository root.
