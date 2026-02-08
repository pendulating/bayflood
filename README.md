# Modeling Urban Street Flooding from Dense Street Imagery

A framework for detecting and analyzing urban street flooding using dashcam imagery, spatial modeling, and multiple data sources. Broadly applies to other urban phenomena visible in public-scene street imagery. 

![High-level process overview of BayFlood.](docs/bayflood_teaser.gif)

### Documentation

- Reproducibility: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
- Data Dependencies: [docs/DATA_DEPENDENCIES.md](docs/DATA_DEPENDENCIES.md)
- Components: [docs/COMPONENTS.md](docs/COMPONENTS.md)
- CLI Reference: [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md)
- Stan Models: [docs/STAN_MODELS.md](docs/STAN_MODELS.md)

## Overview

This repository contains tools and analyses for understanding urban street flooding patterns in New York City using:

- **Zero-shot classification of dense street imagery (here, from networked dashcams)** for automated flood detection
- **ICAR (Intrinsic Conditional Autoregressive) models** for spatial analysis
- **Bayesian inference** using Stan probabilistic programming
- **External sources of flooding**: 311 complaints, FloodNet sensors, census data, topographic data
- **Geospatial analysis** at multiple census geography levels (Census Tracts, Block Groups, Blocks)

## Scope and Key Features

- **Core focus (artifact scope)**: Bayesian spatial modeling (ICAR/CAR) via Stan with `icar_model.py`, tract-level analysis CSVs via `analysis_df.py`, and end-to-end pipeline orchestration via `pipeline.py`.
- **Multi-geometry support**: The pipeline supports Census Tracts (CT), Census Block Groups (CBG), and Census Blocks (CB), configurable via `geometry_config.py`.
- **Out of scope for this artifact**: Submodules `cambrian`, `Janus`, and other external paper repositories (kept as references only).
- **Optional visualization**: `generate_maps.py` can render geospatial maps but is not required for reproducing model outputs.

## Project Structure

```
bayflood/
├── pipeline.py                # End-to-end pipeline (data → model → analysis)
├── icar_model.py              # Main ICAR modeling class
├── geometry_config.py         # Multi-geometry configuration (CT, CBG, CB)
├── util.py                    # Utility functions for data processing
├── generate_maps.py           # Map generation and visualization
├── analysis_df.py             # Analysis DataFrame generation
├── config.py                  # Centralized defaults; env overrides supported
├── logger.py                  # Logging utilities
├── refresh_cache.py           # Cache management
├── stan_models/               # Stan model specifications
│   ├── weighted_ICAR_prior.stan
│   └── ICAR_prior_annotations_have_locations.stan
├── aggregation/               # Data aggregation and processing
│   ├── generate_flooding_dataset.py      # Generate flooding dataset by geometry
│   ├── add_covariates_to_flooding_dataset.py  # Add external covariates
│   ├── aggregate_by_geometry.py          # Parameterized geometry aggregation
│   ├── flooding/              # Flooding data sources (311, FloodNet, DEP)
│   ├── demo/                  # Demographic data (ACS)
│   └── geo/                   # Geographic data and topology processing
├── notebooks/
│   ├── for_paper/             # Original paper analyses
│   ├── for_revisions/         # Revision analyses (see below)
│   └── for_site/              # Website visualizations
├── data/
│   ├── processed/             # Processed datasets
│   ├── revisions/             # Data for revision analyses
│   │   ├── irr/               # Inter-rater reliability data
│   │   ├── nearby_floodnet/   # FloodNet sensor proximity data
│   │   └── prompt_baseline_annotations/  # VLM prompt baseline annotations
│   └── adjacency/             # Adjacency matrices (Stan-compatible format)
├── runs/                      # Model run outputs (FINAL runs included)
└── jobs/                      # SLURM job scripts
```

## Installation

### Prerequisites

- Python 3.10 or higher
- Stan (PyStan)
- A computer with a powerful processor (at least 8 cores), and 64GB of system RAM to run `icar_model.py` at default behavior. RAM requirements increase with the number of model samples.
- The pipeline has been tested on Linux Ubuntu 20.04.

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd bayflood
   ```

2. **Download geospatial data:**
   ```bash
   cd aggregation/geo && bash pull-data.sh && cd ../..
   ```

3. **Create a virtual environment** (use mamba or conda interchangeably):
   ```bash
   mamba create -n bayflood python=3.10
   mamba activate bayflood
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Stan backend**: We use `pystan` for Stan.

## Data Requirements

### Required Data Files

BayFlood utilizes several data sources. GeoJSON boundary files are downloaded via the provided `aggregation/geo/pull-data.sh` script. Other data sources are included in the repository:

1. **Dashcam imagery data** (processed counts per geometry)
2. **Census boundary files** (GeoJSON, downloaded via script)
3. **Demographic data** (ACS 2023)
4. **311 complaint data**
5. **FloodNet sensor data**
6. **Topographic data**

See [docs/DATA_DEPENDENCIES.md](docs/DATA_DEPENDENCIES.md) for a complete listing of required files and their locations.

## Functionality

### End-to-End Pipeline

The `pipeline.py` script runs the complete workflow for any census geometry:

```bash
# Full pipeline for Census Tracts (default)
python pipeline.py --geometry-type ct --prefix my_ct_run --external-covariates

# Full pipeline for Census Block Groups
python pipeline.py --geometry-type cbg --prefix my_cbg_run --external-covariates

# Data generation only (no model fitting)
python pipeline.py --geometry-type cbg --data-only

# Skip data generation (reuse existing processed data)
python pipeline.py --geometry-type ct --prefix rerun --skip-data-generation --external-covariates
```

The pipeline performs these steps:
1. Generate adjacency network (if not exists)
2. Generate topology statistics (if not exists)
3. Generate flooding dataset (image counts per geometry)
4. Add external covariates to the flooding dataset
5. Fit ICAR model
6. Generate maps and analysis

### Individual Components

#### 1. ICAR Model

```python
from icar_model import ICAR_MODEL

model = ICAR_MODEL(
    PREFIX='test_run',
    ICAR_PRIOR_SETTING="icar",
    ANNOTATIONS_HAVE_LOCATIONS=True,
    EXTERNAL_COVARIATES=False,
    SIMULATED_DATA=False,
    ESTIMATE_PARAMS=['p_y', 'at_least_one_positive_image_by_area'],
    EMPIRICAL_DATA_PATH="data/processed/flooding_ct_dataset.csv"
)

model.load_data()
fit = model.fit(CYCLES=1, WARMUP=1000, SAMPLES=1500)
model.plot_results(fit, model.data_to_use)
```

#### 2. Generate Maps

```python
from generate_maps import generate_maps

generate_maps(
    run_id='test_run',
    estimate_path='runs/test_run/estimate_at_least_one_positive_image_by_area.csv',
    estimate='at_least_one_positive_image_by_area'
)
```

#### 3. Analysis DataFrame

```python
from analysis_df import generate_nyc_analysis_df

df = generate_nyc_analysis_df(
    run_dir='runs/test_run',
    custom_prefix='analysis',
    use_smoothing=True
)
```

## Precomputed Runs

Two FINAL model runs are included in the repository for reproducibility:

- **With covariates**: `runs/icar_icar/simulated_False/ahl_True/covariates_True/FINAL_20260206-1100/`
- **Without covariates**: `runs/icar_icar/simulated_False/ahl_True/covariates_False/FINAL_20260206-1205/`

Each run directory contains:
- `analysis_df_FINAL_02062026.csv` — Full tract-level analysis DataFrame
- `analysis_df_describe_FINAL_02062026.csv` — Descriptive statistics
- `metadata.json` — Run configuration and parameters
- `summary.txt` — Stan sampling summary

## Notebooks

### Paper Analyses (`notebooks/for_paper/`)

Analyses from the original paper submission, including coverage maps, bias analyses, downsampled performance, FloodNet placement optimization, and VLM baseline comparisons.

### Revision Analyses (`notebooks/for_revisions/`)

Analyses added during the revision process:

| Notebook | Description |
|----------|-------------|
| `00_hyperlocal_coverage_*.ipynb` | Road-level dashcam coverage analysis |
| `01_power_analysis.ipynb` | Statistical power analysis for flood detection |
| `01a_analysis_external_corrs.ipynb` | Extended external dataset correlations |
| `01b_compare_revisions_vs_paper.ipynb` | Comparison of revision vs. original model outputs |
| `01c_311_biases.ipynb` | 311 complaint reporting bias analysis |
| `01d_analysis_added_coverage.ipynb` | Added coverage analysis |
| `02_downsampled_all_performance.ipynb` | Downsampled annotation performance (all fractions) |
| `03_allexpdays_moremetrics.ipynb` | Multi-day multi-metric evaluation |
| `04_image_sensor_proximity.ipynb` | Dashcam image proximity to FloodNet sensors |
| `05_nearby_sensors_flood_data.ipynb` | FloodNet sensor depth data comparison |
| `06_other_thresholds_coverage_maps.ipynb` | Coverage maps at alternative thresholds |
| `08_prompt_baselines*.ipynb` | VLM prompt engineering baseline comparisons |
| `09_interrater_agreement.ipynb` | Inter-rater reliability (Cohen's kappa) |
| `10_characterizing_misclassified_samples.ipynb` | False positive/negative characterization |
| `adjacency/cbg_geometric_buffer_adjacency.ipynb` | CBG-level adjacency network construction |

Shared utilities: `constants.py`, `helpers.py`, `logger.py`

### External Data Dependencies

A small number of revision notebooks reference external dashcam imagery paths (`/share/ju/nexar_data/`) that are not included in this repository. These notebooks will partially run without the external data but image-level analyses require access to the original imagery.

## Model Specifications

### ICAR Model

The ICAR (Intrinsic Conditional Autoregressive) model accounts for spatial dependencies in flooding patterns:

- **Spatial prior**: ICAR prior on geometry-level flooding probabilities
- **Observation model**: Binomial likelihood for flood detection
- **Covariates**: Optional external covariates (demographics, topography)
- **Inference**: Hamiltonian Monte Carlo via Stan

### Stan Models

Located in `stan_models/`:
- `ICAR_prior_annotations_have_locations.stan`: ICAR model with annotation locations (primary model used in paper)
- `weighted_ICAR_prior.stan`: ICAR model without annotation location data

## Outputs

### Model Outputs

- **Parameter estimates**: CSV files with posterior means and intervals
- **Diagnostic plots**: Convergence diagnostics, posterior distributions
- **Spatial maps**: Geographic visualizations of flooding risk

### Analysis Outputs

- **DataFrames**: Combined analysis with all covariates, plus descriptive statistics
- **Statistical summaries**: Correlation analyses, bias assessments
- **Visualizations**: Maps and plots

## Citation

If you use or build off of this work, please cite:

- Bayesian Modeling of Zero-Shot Classifications for Urban Flood Detection. arXiv:2503.14754v2, 26 Mar 2025. [arXiv](https://arxiv.org/abs/2503.14754v2)

This repository includes a `CITATION.cff` (use GitHub's "Cite this repository" for formatted citations).


## Contact

For questions or issues, please open a GitHub issue or contact mattfranchi@cs.cornell.edu

## Acknowledgments
We thank Gabriel Agostini, Sidhika Balachandar, Serina Chang, Zhi Liu, and Anna McClendon for useful discussion and feedback. We thank Nexar for data access under research evaluation and project support. We thank Anthony Townsend and Michael Samuelian for project support. We thank the NYC Department of Environmental Protection for helpful discussions. We thank Charlie Mydlarz and the FloodNet team for helpful discussions and access to FloodNet data. We thank OpenAI for LLM inference credits. We thank the Digital Life Initiative, the Urban Tech Hub at Cornell Tech, a Google Research Scholar award, an AI2050 Early Career Fellowship, NSF CAREER \#2142419, NSF CAREER IIS-2339427, a CIFAR Azrieli Global scholarship, a gift to the LinkedIn-Cornell Bowers CIS Strategic Partnership, the Survival and Flourishing Fund, and the Abby Joseph Cohen Faculty Fund for funding. 
