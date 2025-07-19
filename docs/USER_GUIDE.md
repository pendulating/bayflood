# User Guide: Reproducing Results and Conducting Analysis

This guide provides detailed instructions for reproducing the main results from the BayFlood analysis and conducting your own analyses.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Data Preparation](#data-preparation)
3. [Reproducing Main Results](#reproducing-main-results)
4. [Conducting Custom Analyses](#conducting-custom-analyses)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

Before starting, ensure you have:

- ✅ Completed the [Installation Guide](INSTALLATION.md)
- ✅ Completed the [Quick Start Guide](QUICKSTART.md)
- ✅ All required data files (see Data Requirements below)
- ✅ Sufficient computational resources (16GB+ RAM recommended)

## Data Preparation

### Required Data Files

The analysis requires the following data files organized in the correct directory structure:

```
data/
├── processed/
│   ├── flooding_ct_dataset.csv          # Main flooding dataset
│   ├── sep29_positives.csv              # September 29 positive images
│   ├── sep29_gt.csv                     # Ground truth annotations
│   └── md.csv                           # Image metadata
├── raw/                                 # Raw data files
└── external/                            # External data sources

aggregation/
├── flooding/
│   ├── data/
│   │   ├── nyc311_flooding_sep29.csv    # 311 complaints
│   │   └── NYCFloodStormwaterFloodMaps/ # DEP flood maps
│   └── static/
│       ├── floodnet_sensor_coordinates.csv
│       ├── floodnet-flood-sensor-sep-2023.csv
│       └── current_floodnet_sensors.csv
├── demo/
│   └── data/
│       ├── acs2023_dp05.json            # Race/demographics
│       ├── acs2023_s2801.json           # Internet access
│       ├── acs2023_s1901.json           # Income
│       ├── acs2023_s1501.json           # Education
│       └── acs2023_s1602.json           # Language
└── geo/
    ├── data/
    │   ├── ct-nyc-2020.geojson          # Census tract boundaries
    │   └── processed/
    │       └── ct_nyc_topology.csv      # Topographic data
    └── adjacency/                       # Spatial adjacency files
```

### Data Format Requirements

#### Main Dataset (`flooding_ct_dataset.csv`)

Required columns:
- `GEOID`: Census tract identifier (string)
- `n_total`: Total number of images per tract (integer)
- `n_classified_positive`: Number of flood-positive images (integer)
- `geometry`: Tract geometry in WKT format (string)

Optional columns (if `annotations_have_locations=True`):
- `n_tp`: True positive annotations (integer)
- `n_fn`: False negative annotations (integer)
- `n_fp`: False positive annotations (integer)
- `n_tn`: True negative annotations (integer)
- `total_not_annotated`: Unannotated images (integer)
- `positives_not_annotated`: Unannotated positive images (integer)

#### Census Tract Boundaries (`ct-nyc-2020.geojson`)

Standard GeoJSON format with:
- `GEOID`: Tract identifier
- `geometry`: Polygon geometry
- Additional metadata columns

#### Demographic Data (ACS JSON files)

JSON files with ACS 2023 data containing:
- Variable codes and descriptions
- Estimates for NYC census tracts
- Proper geographic identifiers

## Reproducing Main Results

### 1. Basic ICAR Model Results

Reproduce the main flooding probability estimates:

```python
from icar_model import ICAR_MODEL

# Initialize model with paper settings
model = ICAR_MODEL(
    PREFIX='paper_reproduction',
    ICAR_PRIOR_SETTING="icar",
    ANNOTATIONS_HAVE_LOCATIONS=True,
    EXTERNAL_COVARIATES=True,
    SIMULATED_DATA=False,
    ESTIMATE_PARAMS=['p_y', 'at_least_one_positive_image_by_area'],
    EMPIRICAL_DATA_PATH="data/processed/flooding_ct_dataset.csv"
)

# Load data
model.load_data()

# Fit model with paper parameters
fit = model.fit(CYCLES=1, WARMUP=1000, SAMPLES=1500)

# Generate results
model.plot_results(fit, model.data_to_use)
model.write_estimate(fit, model.data_to_use)
```

### 2. Generate Main Figures

#### Flooding Probability Map

```python
from generate_maps import generate_maps

# Generate main flooding map
generate_maps(
    run_id='paper_reproduction',
    estimate_path='runs/paper_reproduction/estimate_at_least_one_positive_image_by_area.csv',
    estimate='at_least_one_positive_image_by_area'
)
```

#### Comprehensive Analysis

```python
from analysis_df import generate_nyc_analysis_df

# Generate comprehensive analysis DataFrame
analysis_df = generate_nyc_analysis_df(
    run_dir='runs/paper_reproduction',
    custom_prefix='paper_analysis',
    use_smoothing=True
)

# Save results
analysis_df.to_csv('deliverables/paper_analysis_results.csv', index=False)
```

### 3. Reproduce Specific Paper Figures

#### Figure 1: Flooding Distribution Map

```python
# Use the notebook: notebooks/for_paper/f_dashcam_distribution_map.ipynb
# This reproduces the main flooding distribution visualization
```

#### Figure 2: Sensor Placement Analysis

```python
# Use the notebook: notebooks/for_paper/f_floodnet_placement.ipynb
# This reproduces the FloodNet sensor placement analysis
```

#### Figure 3: Bias Analysis

```python
# Use the notebook: notebooks/for_paper/f_311_biases.ipynb
# This reproduces the 311 complaint bias analysis
```

## Conducting Custom Analyses

### 1. Custom ICAR Model Configuration

#### Different Prior Specifications

```python
# Uniform prior (no spatial structure)
model_uniform = ICAR_MODEL(
    PREFIX='uniform_prior',
    ICAR_PRIOR_SETTING="none",
    ANNOTATIONS_HAVE_LOCATIONS=True,
    EXTERNAL_COVARIATES=False,
    ESTIMATE_PARAMS=['p_y'],
    EMPIRICAL_DATA_PATH="data/processed/flooding_ct_dataset.csv"
)

# Proper CAR prior
model_proper = ICAR_MODEL(
    PREFIX='proper_car',
    ICAR_PRIOR_SETTING="proper",
    ANNOTATIONS_HAVE_LOCATIONS=True,
    EXTERNAL_COVARIATES=True,
    ESTIMATE_PARAMS=['p_y', 'at_least_one_positive_image_by_area'],
    EMPIRICAL_DATA_PATH="data/processed/flooding_ct_dataset.csv"
)
```

#### Custom Adjacency Matrix

```python
# Use custom adjacency matrix
model_custom_adj = ICAR_MODEL(
    PREFIX='custom_adjacency',
    ICAR_PRIOR_SETTING="icar",
    ANNOTATIONS_HAVE_LOCATIONS=True,
    EXTERNAL_COVARIATES=False,
    ESTIMATE_PARAMS=['p_y'],
    EMPIRICAL_DATA_PATH="data/processed/flooding_ct_dataset.csv",
    adj=['data/adjacency/node1.txt', 'data/adjacency/node2.txt'],
    adj_matrix_storage=False
)
```

### 2. Sensitivity Analysis

#### Parameter Sensitivity

```python
# Test different warmup and sample sizes
configurations = [
    {'WARMUP': 500, 'SAMPLES': 1000},
    {'WARMUP': 1000, 'SAMPLES': 1500},
    {'WARMUP': 2000, 'SAMPLES': 3000}
]

results = {}
for i, config in enumerate(configurations):
    model = ICAR_MODEL(
        PREFIX=f'sensitivity_{i}',
        ICAR_PRIOR_SETTING="icar",
        ANNOTATIONS_HAVE_LOCATIONS=True,
        EXTERNAL_COVARIATES=True,
        ESTIMATE_PARAMS=['p_y'],
        EMPIRICAL_DATA_PATH="data/processed/flooding_ct_dataset.csv"
    )
    model.load_data()
    fit = model.fit(**config)
    results[f'config_{i}'] = fit
```

#### Data Sensitivity

```python
# Test with different downsampling fractions
fractions = [0.1, 0.25, 0.5, 1.0]

for frac in fractions:
    model = ICAR_MODEL(
        PREFIX=f'downsample_{frac}',
        ICAR_PRIOR_SETTING="icar",
        ANNOTATIONS_HAVE_LOCATIONS=True,
        EXTERNAL_COVARIATES=True,
        ESTIMATE_PARAMS=['p_y'],
        EMPIRICAL_DATA_PATH="data/processed/flooding_ct_dataset.csv",
        downsample_frac=frac
    )
    model.load_data()
    fit = model.fit(CYCLES=1, WARMUP=1000, SAMPLES=1500)
```

### 3. Comparative Analysis

#### Model Comparison

```python
# Compare different model specifications
models_to_compare = {
    'uniform': {'ICAR_PRIOR_SETTING': 'none', 'EXTERNAL_COVARIATES': False},
    'icar_only': {'ICAR_PRIOR_SETTING': 'icar', 'EXTERNAL_COVARIATES': False},
    'icar_covariates': {'ICAR_PRIOR_SETTING': 'icar', 'EXTERNAL_COVARIATES': True},
    'proper_car': {'ICAR_PRIOR_SETTING': 'proper', 'EXTERNAL_COVARIATES': True}
}

comparison_results = {}
for name, config in models_to_compare.items():
    model = ICAR_MODEL(
        PREFIX=f'comparison_{name}',
        ANNOTATIONS_HAVE_LOCATIONS=True,
        ESTIMATE_PARAMS=['p_y'],
        EMPIRICAL_DATA_PATH="data/processed/flooding_ct_dataset.csv",
        **config
    )
    model.load_data()
    fit = model.fit(CYCLES=1, WARMUP=1000, SAMPLES=1500)
    comparison_results[name] = fit
```

### 4. Cross-Validation

```python
# Perform cross-validation
model = ICAR_MODEL(
    PREFIX='cross_validation',
    ICAR_PRIOR_SETTING="icar",
    ANNOTATIONS_HAVE_LOCATIONS=True,
    EXTERNAL_COVARIATES=True,
    ESTIMATE_PARAMS=['p_y'],
    EMPIRICAL_DATA_PATH="data/processed/flooding_ct_dataset.csv"
)

# Compare to baselines
model.compare_to_baselines(train_frac=0.2, save=True)
```

## Advanced Features

### 1. Custom Stan Models

Create custom Stan model specifications:

```python
# Load custom Stan model
with open('stan_models/custom_model.stan', 'r') as f:
    custom_model_code = f.read()

# Add to model dictionary
model.models['custom'] = custom_model_code

# Use custom model
model.icar_prior_setting = 'custom'
```

### 2. External Covariates Analysis

```python
# Analyze external covariates
model = ICAR_MODEL(
    PREFIX='covariates_analysis',
    ICAR_PRIOR_SETTING="icar",
    ANNOTATIONS_HAVE_LOCATIONS=True,
    EXTERNAL_COVARIATES=True,
    ESTIMATE_PARAMS=['p_y'],
    EMPIRICAL_DATA_PATH="data/processed/flooding_ct_dataset.csv"
)

model.load_data()
fit = model.fit(CYCLES=1, WARMUP=1000, SAMPLES=1500)

# Analyze covariate effects
summary = fit.summary()
covariate_effects = summary[summary.index.str.contains('beta')]
print("Covariate effects:")
print(covariate_effects)
```

### 3. Spatial Analysis

```python
# Analyze spatial patterns
from analysis_df import generate_nyc_analysis_df

analysis_df = generate_nyc_analysis_df(
    run_dir='runs/paper_reproduction',
    custom_prefix='spatial_analysis',
    use_smoothing=True
)

# Spatial autocorrelation
import geopandas as gpd
from libpysal.weights import Queen
from libpysal.explore.esda import Moran

# Create spatial weights
gdf = gpd.GeoDataFrame(analysis_df, geometry='geometry')
weights = Queen.from_dataframe(gdf)

# Calculate Moran's I
moran = Moran(analysis_df['estimate_p_y'], weights)
print(f"Moran's I: {moran.I}")
print(f"P-value: {moran.p_norm}")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Model Convergence Issues

**Problem**: Stan model fails to converge

**Solutions**:
```python
# Increase warmup and samples
fit = model.fit(CYCLES=1, WARMUP=2000, SAMPLES=3000)

# Check convergence diagnostics
summary = fit.summary()
print(summary['Rhat'].max())  # Should be < 1.1

# Use different prior specifications
model.icar_prior_setting = "proper"
```

#### 2. Memory Issues

**Problem**: Out of memory errors

**Solutions**:
```python
# Use downsampling
model = ICAR_MODEL(..., downsample_frac=0.5)

# Reduce sample size
fit = model.fit(CYCLES=1, WARMUP=500, SAMPLES=1000)

# Use chunked processing for large datasets
```

#### 3. Data Format Issues

**Problem**: Data loading errors

**Solutions**:
```python
# Validate data format
import pandas as pd
df = pd.read_csv("data/processed/flooding_ct_dataset.csv")
print("Required columns:", ['GEOID', 'n_total', 'n_classified_positive'])
print("Available columns:", list(df.columns))

# Check data types
print(df.dtypes)

# Handle missing values
df = df.fillna(0)
```

#### 4. Stan Compilation Issues

**Problem**: Stan model compilation fails

**Solutions**:
```python
# Check Stan installation
import cmdstanpy
print(f"Stan version: {cmdstanpy.__version__}")

# Reinstall Stan
import cmdstanpy
cmdstanpy.install_cmdstan()

# Check model syntax
model_code = model.models[model.icar_prior_setting]
# Validate Stan syntax manually
```

### Performance Optimization

#### 1. Parallel Processing

```python
# Set number of threads
import os
os.environ['STAN_NUM_THREADS'] = '4'

# Use parallel chains
fit = model.fit(CYCLES=4, WARMUP=1000, SAMPLES=1500)
```

#### 2. Memory Management

```python
# Clear memory between runs
import gc
gc.collect()

# Use memory-efficient data types
df = df.astype({'GEOID': 'string', 'n_total': 'int32', 'n_classified_positive': 'int32'})
```

## Next Steps

After completing your analysis:

1. **Document your results**: Create clear documentation of findings
2. **Share your code**: Use version control and share reproducible code
3. **Validate results**: Cross-check with different methods
4. **Contribute**: Submit improvements to the codebase

## Getting Help

- **Documentation**: Check the `docs/` folder for detailed guides
- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions
- **Email**: Contact the development team for specific issues 