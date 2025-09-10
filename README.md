# Modeling Urban Street Flooding from Dense Street Imagery

A comprehensive analysis framework for detecting and analyzing urban street flooding using dashcam imagery, spatial modeling, and multiple data sources.

![High-level process overview of BayFlood.](docs/bayflood_teaser.gif)

## Overview

This repository contains tools and analyses for understanding urban street flooding patterns in New York City using:

- **Zero-shot classification of dense street imagery (here, from networked dashcams)** for automated flood detection
- **ICAR (Intrinsic Conditional Autoregressive) models** for spatial analysis
- **Bayesian inference** using Stan probabilistic programming
- **External sources of flooding**: 311 complaints, FloodNet sensors, census data, topographic data
- **Geospatial analysis** with NYC census tracts as the primary unit

## Scope and Key Features

- **Core focus (artifact scope)**: Bayesian spatial modeling (ICAR/CAR) via Stan with `icar_model.py`, and tract-level analysis CSVs via `analysis_df.py`.
- **Out of scope for this artifact**: Submodules `cambrian`, `Janus`, and other external paper repositories (kept as references only).
- **Optional visualization**: `generate_maps.py` can render geospatial maps but is not required for reproducing model outputs.

## Project Structure (relevant to ICAR pipeline)

```
bayflood/
├── icar_model.py              # Main ICAR modeling class
├── util.py                    # Utility functions for data processing
├── generate_maps.py           # Map generation and visualization
├── analysis_df.py             # Analysis DataFrame generation
├── logger.py                  # Logging utilities
├── refresh_cache.py           # Cache management
├── config.py                  # Centralized defaults; env overrides supported
├── observed_data.csv          # Processed flooding observations
├── stan_models/               # Stan model specifications
│   ├── weighted_ICAR_prior.stan
│   ├── ICAR_prior_annotations_have_locations.stan
├── notebooks/                 # Jupyter notebooks for analysis
│   ├── for_paper/            # Paper-specific analyses
│   └── visual_assets
├── data/                      # Data storage
│   ├── processed/            # Processed datasets
│   └── adjacency/            # Pre-computed adjacency matrix of NYC census tracts, in Stan-compatible format
├── aggregation/              # Aggregated data sources
│   ├── flooding/            # Flooding-related data
│   ├── demo/                # Demographic data
│   └── geo/                 # Geographic data
└── runs/                     # Model run outputs (two replication runs included in Repo)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Stan (PyStan)

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd bayflood
   ```

2. **Create a virtual environment:**
   ```bash
   conda create -n bayflood python=3.10
   conda activate bayflood
   ```

3. **Install dependencies (Python 3.10):**
   ```bash
   pip install -r requirements.txt  # or: pip install -r requirements-core.txt
   ```

   Or install manually:
   ```bash
   pip install pandas numpy scipy scikit-learn
   pip install geopandas matplotlib seaborn
   pip install stan pystan arviz
   pip install jupyter notebook
   pip install shapely pyproj
   ```

4. **Stan backend**: We use `pystan` for Stan.

## Data Requirements

### Required Data Files

The analysis requires several data sources:

1. **Dashcam imagery data** (processed)
2. **Census tract boundaries** (GeoJSON format)
3. **Demographic data** (ACS 2023)
4. **311 complaint data**
5. **FloodNet sensor data**
6. **Topographic data**

### Data Organization

Place data files in the appropriate directories:
- Raw data: `data/`
- Processed data: `data/processed/`
- Aggregated data: `aggregation/`

## Quick Start

### 1. Basic ICAR Model Usage

```python
from icar_model import ICAR_MODEL

# Initialize model
model = ICAR_MODEL(
    PREFIX='test_run',
    ICAR_PRIOR_SETTING="icar",
    ANNOTATIONS_HAVE_LOCATIONS=True,
    EXTERNAL_COVARIATES=False,
    SIMULATED_DATA=False,
    ESTIMATE_PARAMS=['p_y', 'at_least_one_positive_image_by_area'],
    EMPIRICAL_DATA_PATH="data/processed/flooding_ct_dataset.csv"
)

# Load data
model.load_data()

# Fit model
fit = model.fit(CYCLES=1, WARMUP=1000, SAMPLES=1500)

# Generate results
model.plot_results(fit, model.data_to_use)
```

### 2. Generate Maps

```python
from generate_maps import generate_maps

# Generate flooding maps
generate_maps(
    run_id='test_run',
    estimate_path='runs/test_run/estimate_at_least_one_positive_image_by_area.csv',
    estimate='at_least_one_positive_image_by_area'
)
```

### 3. Analysis DataFrame

```python
from analysis_df import generate_nyc_analysis_df

# Generate comprehensive analysis
df = generate_nyc_analysis_df(
    run_dir='runs/test_run',
    custom_prefix='analysis',
    use_smoothing=True
)
```

## Usage Examples

### Running a Complete Analysis

1. **Prepare your data** according to the data requirements
2. **Configure model parameters** via CLI flags or environment variables in `config.py`
3. **Run the ICAR model** to get flooding estimates
4. **Generate visualizations** using `generate_maps.py`
5. **Perform additional analysis** using the notebooks

## End-to-end usage example (conda env → train → maps → analysis)

- Create and activate a fresh conda environment (Python 3.10)
```bash
conda create -n bayflood-icar python=3.10 -y
conda activate bayflood-icar
```

- Install dependencies (recommended: conda for geo libs, pip for the rest)
```bash
# Core + geospatial via conda-forge
conda install -c conda-forge numpy scipy pandas scikit-learn matplotlib seaborn jupyter -y
conda install -c conda-forge geopandas shapely pyproj fiona rasterio pyarrow -y

# Stan http backend + utils via pip
pip install stan arviz nest-asyncio rasterstats tqdm python-json-logger termcolor
```
Alternatively:
```bash
pip install -r requirements-core.txt
```

- Move into the repo and (optional) clear Stan cache
```bash
cd /share/ju/matt/street-flooding
python -c "from refresh_cache import refresh_cache; refresh_cache()"
```

- Verify required data are present (adjust paths as needed)
```bash
ls aggregation/context_df_02102025.csv
ls aggregation/geo/data/ct-nyc-2020.geojson
ls aggregation/flooding/data/nyc311_flooding_sep29.csv
ls aggregation/flooding/static/current_floodnet_sensors.csv
# Optional for maps depending on local data layout
# ls "aggregation/flooding/data/NYCFloodStormwaterFloodMaps/NYC Stormwater Flood Map - Moderate Flood (2.13 inches per hr) with Current Sea Levels/NYC_Stormwater_Flood_Map_Moderate_Flood_2_13_inches_per_hr_with_Current_Sea_Levels.gdb"
```

- Train a new ICAR model on the provided dataset (with covariates)
```bash
EMPIRICAL="aggregation/context_df_02102025.csv"

python icar_model.py icar \
  --annotations_have_locations \
  --external_covariates \
  --prefix VALIDATION_WITH_COVS \
  --empirical_data_path "$EMPIRICAL"
```

- (Optional) Train without covariates for comparison
```bash
python icar_model.py icar \
  --annotations_have_locations \
  --prefix VALIDATION_NO_COVS \
  --empirical_data_path "$EMPIRICAL"
```

- Locate the latest run ID (with covariates)
```bash
RUN_DIR=$(ls -td runs/icar_icar/simulated_False/ahl_True/covariates_True/* | head -1)
RUN_ID=${RUN_DIR#runs/}
echo "$RUN_ID"
```

- Generate maps from the new run (optional)
```bash
python generate_maps.py "$RUN_ID" "runs/$RUN_ID/estimate_p_y.csv" p_y
python generate_maps.py "$RUN_ID" "runs/$RUN_ID/estimate_at_least_one_positive_image_by_area.csv" at_least_one_positive_image_by_area
```

- Generate the tract-level analysis CSVs (core output)
```bash
python -c "from analysis_df import generate_nyc_analysis_df as g; g(run_dir='runs/$RUN_ID', custom_prefix='validation', use_smoothing=True)"
```

- Validate outputs exist
```bash
ls runs/$RUN_ID/estimate_p_y.csv
ls runs/$RUN_ID/analysis_df_validation_*.csv
ls runs/$RUN_ID/analysis_df_describe_validation_*.csv
```

### Notebooks

Paper notebooks live in submodules and are out of scope for this artifact.

## Model Specifications

### ICAR Model

The ICAR (Intrinsic Conditional Autoregressive) model accounts for spatial dependencies in flooding patterns:

- **Spatial prior**: ICAR prior on tract-level flooding probabilities
- **Observation model**: Binomial likelihood for flood detection
- **Covariates**: Optional external covariates (demographics, topography)
- **Inference**: Hamiltonian Monte Carlo via Stan

### Stan Models

Located in `stan_models/`:
- `ICAR_prior_annotations_have_locations.stan`: ICAR model with annotation locations (only model used)

## Outputs

### Model Outputs

- **Parameter estimates**: CSV files with posterior means and intervals
- **Diagnostic plots**: Convergence diagnostics, posterior distributions
- **Spatial maps**: Geographic visualizations of flooding risk

### Analysis Outputs

- **Comprehensive DataFrames**: Combined analysis with all covariates
- **Statistical summaries**: Correlation analyses, bias assessments
- **Visualizations**: Maps, plots, and interactive figures

## Citation

If you use this work, please cite:

- Bayesian Modeling of Zero-Shot Classifications for Urban Flood Detection. arXiv:2503.14754v2, 26 Mar 2025. [arXiv](https://arxiv.org/abs/2503.14754v2)

This repository includes a `CITATION.cff` (use GitHub’s “Cite this repository” for formatted citations).

## License

Add a `LICENSE` file at the repository root.

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

## Acknowledgments

- [List any acknowledgments, funding sources, etc.]
