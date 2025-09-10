## Reproducibility Guide (ICAR Artifact Scope)

This guide explains how to reproduce the main results using precomputed run outputs and how to regenerate analyses from those runs.

### 1) Precomputed runs (examples)

Example precomputed runs are included at:
- Without covariates:
  `runs/icar_icar/simulated_False/ahl_True/covariates_False/FEB7_FINAL_KDD_MODEL_NO_COVARIATES_20250207-1814`
- With covariates:
  `runs/icar_icar/simulated_False/ahl_True/covariates_True/FEB7_FINAL_KDD_MODEL_20250207-1732`

Each run directory contains at least:
- `estimate_p_y.csv`
- `estimate_at_least_one_positive_image_by_area.csv`
- `summary.txt`
- `metadata.json`

### 2) Generate tract-level analysis CSVs from a run

```python
from analysis_df import generate_nyc_analysis_df

# With covariates
df_cov = generate_nyc_analysis_df(
    run_dir='runs/icar_icar/simulated_False/ahl_True/covariates_True/FEB7_FINAL_KDD_MODEL_20250207-1732',
    custom_prefix='with_covariates',
    use_smoothing=True
)

# Without covariates
df_nocov = generate_nyc_analysis_df(
    run_dir='runs/icar_icar/simulated_False/ahl_True/covariates_False/FEB7_FINAL_KDD_MODEL_NO_COVARIATES_20250207-1814',
    custom_prefix='without_covariates',
    use_smoothing=True
)
```

Outputs are written back to each `run_dir` (e.g., `analysis_df_*.csv` and `analysis_df_describe_*.csv`).

### 3) (Optional) Generate maps from a run

`generate_maps.py` is optional.

```python
from generate_maps import generate_maps

# p(y)
generate_maps(
    run_id='icar_icar/simulated_False/ahl_True/covariates_True/FEB7_FINAL_KDD_MODEL_20250207-1732',
    estimate_path='runs/icar_icar/simulated_False/ahl_True/covariates_True/FEB7_FINAL_KDD_MODEL_20250207-1732/estimate_p_y.csv',
    estimate='p_y'
)

# at_least_one_positive_image_by_area
generate_maps(
    run_id='icar_icar/simulated_False/ahl_True/covariates_True/FEB7_FINAL_KDD_MODEL_20250207-1732',
    estimate_path='runs/icar_icar/simulated_False/ahl_True/covariates_True/FEB7_FINAL_KDD_MODEL_20250207-1732/estimate_at_least_one_positive_image_by_area.csv',
    estimate='at_least_one_positive_image_by_area'
)
```

### 4) Regenerating runs (optional)

To re-run training (requires a powerful computer):

```bash
python icar_model.py icar \
  --annotations_have_locations \
  --external_covariates \
  --prefix REPRO_REDO \
  --empirical_data_path data/processed/flooding_ct_dataset.csv \
  --adj_node1_path data/adjacency/cg_500/ct_nyc_adj_list_custom_geometric_node1.txt \
  --adj_node2_path data/adjacency/cg_500/ct_nyc_adj_list_custom_geometric_node2.txt
```

Then run the analysis step on the new run directory as in step (2).

### Notes
- Python 3.10 and the `pystan` backend are used.
- See `docs/DATA_DEPENDENCIES.md` for required datasets and expected paths.
- For exact citation and licensing, add `CITATION.cff` and `LICENSE` at the repository root.
