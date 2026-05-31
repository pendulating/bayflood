# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

BayFlood is a Bayesian spatial modeling framework for detecting urban street flooding from dense dashcam imagery in NYC. It uses ICAR (Intrinsic Conditional Autoregressive) models in Stan across multiple census geography levels (Census Tracts, Block Groups, Blocks).

## Key Commands

### Environment Setup
```bash
mamba create -n bayflood python=3.10
mamba activate bayflood
pip install -r requirements.txt
# Download geospatial boundary files:
cd aggregation/geo && bash pull-data.sh && cd ../..
```

### Running the Pipeline
```bash
# Full end-to-end pipeline (recommended entry point)
python pipeline.py --geometry-type ct --prefix my_run --external-covariates

# Data generation only (no model fitting)
python pipeline.py --geometry-type cbg --data-only

# Skip data generation, reuse existing processed data
python pipeline.py --geometry-type ct --prefix rerun --skip-data-generation --external-covariates

# Direct model fitting
python icar_model.py icar --annotations_have_locations --external_covariates --geometry_type ct
```

### Linting/Formatting
```bash
black .
flake8 .
```

### Tests
```bash
pytest
```

## Architecture

### Data Flow
```
Raw dashcam predictions
  → generate_flooding_dataset.py (aggregate counts per geometry)
  → add_covariates_to_flooding_dataset.py (add topology, 311, FloodNet, DEP)
  → icar_model.py (Bayesian ICAR inference via Stan/PyStan)
  → analysis_df.py (merge estimates with ACS demographics)
```

### Multi-Geometry System
Three census levels are supported: CT (Census Tracts), CBG (Block Groups), CB (Blocks). The system is configured through:
- `geometry_config.py` — `GeometryType` enum, `GeometryConfig` dataclass, `GeometryPaths` path factory
- `config.py` — environment variable overrides for paths and sampling parameters
- CLI `--geometry-type` flags on `pipeline.py` and `icar_model.py`

### Core Modules
- **`pipeline.py`** — End-to-end orchestration (uses `fire` CLI). Runs: adjacency generation → topology → dataset → covariates → model → maps → analysis.
- **`icar_model.py`** — `ICAR_MODEL` class (~1600 lines). Handles data loading, Stan compilation, MCMC fitting, posterior extraction, baseline comparisons. Uses `argparse` CLI.
- **`util.py`** — Data I/O (`read_real_data`), adjacency construction, covariate matrix engineering, simulated data generation.
- **`geometry_config.py`** — `GeometryType` enum, `GeometryConfig` dataclass, `GeometryPaths` class for geometry-specific file paths.
- **`config.py`** — Centralized defaults with env var overrides (`BAYFLOOD_GEOMETRY_TYPE`, `EMPIRICAL_DATA_PATH`, `DEFAULT_WARMUP`, `DEFAULT_SAMPLES`, etc.).
- **`analysis_df.py`** — Merges model estimates with covariates for final analysis CSVs.

### Stan Models
Located in `stan_models/`. Primary model: `ICAR_prior_annotations_have_locations.stan`. Uses PyStan (not cmdstanpy). Stan cache is stored in `.cache/` (project-local). aiohttp timeout is patched to 30min for large geometries like CBG (~6800 areas).

### Aggregation Subsystem (`aggregation/`)
- `geo/` — GeoJSON boundaries, topology processing, adjacency network generation
- `flooding/` — External flood data sources (311, FloodNet sensors, DEP catch basins)
- `demo/` — ACS demographic data

### Output Structure
Model runs write to `runs/icar_icar/simulated_False/ahl_True/covariates_True/<PREFIX>_<TIMESTAMP>/` containing estimate CSVs, analysis DataFrames, metadata JSON, and summary text.

## System Requirements
- Python 3.10+, PyStan 3.7+
- 8+ cores and 64GB RAM recommended for model fitting
- Tested on Linux Ubuntu 20.04
- SLURM job scripts in `jobs/` for cluster execution

## Documentation
Detailed docs in `docs/`: REPRODUCIBILITY.md, COMPONENTS.md, DATA_DEPENDENCIES.md, CLI_REFERENCE.md, STAN_MODELS.md.
