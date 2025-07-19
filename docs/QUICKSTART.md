# Quick Start Guide

This guide will help you get started with BayFlood in under 30 minutes.

## Prerequisites

Before starting, ensure you have:
- ✅ Completed the [Installation Guide](INSTALLATION.md)
- ✅ Python environment activated
- ✅ Required data files (see Data Requirements below)

## Data Requirements

### Minimum Required Data

For a basic analysis, you need:

1. **Processed flooding dataset**: `data/processed/flooding_ct_dataset.csv`
2. **Census tract boundaries**: `aggregation/geo/data/ct-nyc-2020.geojson`
3. **Basic demographic data**: ACS 2023 JSON files in `aggregation/demo/data/`

### Data Structure

Your `flooding_ct_dataset.csv` should contain:
- `GEOID`: Census tract identifier
- `n_total`: Total number of images per tract
- `n_classified_positive`: Number of flood-positive images
- `geometry`: Tract geometry (WKT format)

## Quick Start: Basic Analysis

### Step 1: Verify Installation

```python
# Test imports
import pandas as pd
import numpy as np
import geopandas as gpd
import stan
import matplotlib.pyplot as plt

print("✅ All dependencies imported successfully!")
```

### Step 2: Load and Explore Data

```python
# Load the main dataset
df = pd.read_csv("data/processed/flooding_ct_dataset.csv")
print(f"Dataset shape: {df.shape}")
print(f"Number of tracts: {len(df)}")
print(f"Total images: {df['n_total'].sum()}")
print(f"Positive images: {df['n_classified_positive'].sum()}")
```

### Step 3: Run Basic ICAR Model

```python
from icar_model import ICAR_MODEL

# Initialize model with basic settings
model = ICAR_MODEL(
    PREFIX='quickstart',
    ICAR_PRIOR_SETTING="icar",
    ANNOTATIONS_HAVE_LOCATIONS=False,  # Set to True if you have location data
    EXTERNAL_COVARIATES=False,
    SIMULATED_DATA=False,
    ESTIMATE_PARAMS=['p_y'],
    EMPIRICAL_DATA_PATH="data/processed/flooding_ct_dataset.csv"
)

# Load data
print("Loading data...")
model.load_data()

# Fit model (reduced samples for quick start)
print("Fitting model...")
fit = model.fit(CYCLES=1, WARMUP=500, SAMPLES=1000)

print("✅ Model fitting complete!")
```

### Step 4: Generate Basic Results

```python
# Plot results
model.plot_results(fit, model.data_to_use)

# Save estimates
model.write_estimate(fit, model.data_to_use)
```

### Step 5: Create Basic Map

```python
from generate_maps import generate_maps

# Generate flooding map
generate_maps(
    run_id='quickstart',
    estimate_path='runs/quickstart/estimate_p_y.csv',
    estimate='p_y'
)

print("✅ Map generated! Check the deliverables folder.")
```

## Quick Start: Advanced Analysis

### Step 1: Full Model with External Covariates

```python
# Initialize model with external covariates
model_advanced = ICAR_MODEL(
    PREFIX='advanced',
    ICAR_PRIOR_SETTING="icar",
    ANNOTATIONS_HAVE_LOCATIONS=True,
    EXTERNAL_COVARIATES=True,
    SIMULATED_DATA=False,
    ESTIMATE_PARAMS=['p_y', 'at_least_one_positive_image_by_area'],
    EMPIRICAL_DATA_PATH="data/processed/flooding_ct_dataset.csv"
)

# Load and fit
model_advanced.load_data()
fit_advanced = model_advanced.fit(CYCLES=1, WARMUP=1000, SAMPLES=1500)
```

### Step 2: Comprehensive Analysis

```python
from analysis_df import generate_nyc_analysis_df

# Generate comprehensive analysis DataFrame
analysis_df = generate_nyc_analysis_df(
    run_dir='runs/advanced',
    custom_prefix='comprehensive',
    use_smoothing=True
)

print(f"Analysis DataFrame shape: {analysis_df.shape}")
print(f"Columns: {list(analysis_df.columns)}")
```

### Step 3: Explore Results

```python
# Basic statistics
print("Flooding probability summary:")
print(analysis_df['estimate_p_y'].describe())

# Correlation with demographics
correlations = analysis_df[['estimate_p_y', 'median_household_income', 'total_population']].corr()
print("\nCorrelations:")
print(correlations)
```

## Example Notebooks

### Basic Analysis Notebook

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/for_paper/f_basic_analysis.ipynb
```

### Visualization Notebook

```bash
# Open: notebooks/visualization/basic_maps.ipynb
```

## Common Quick Start Scenarios

### Scenario 1: Reproduce Paper Results

```python
# Use the exact settings from the paper
model_paper = ICAR_MODEL(
    PREFIX='paper_reproduction',
    ICAR_PRIOR_SETTING="icar",
    ANNOTATIONS_HAVE_LOCATIONS=True,
    EXTERNAL_COVARIATES=True,
    ESTIMATE_PARAMS=['p_y', 'at_least_one_positive_image_by_area'],
    EMPIRICAL_DATA_PATH="data/processed/flooding_ct_dataset.csv"
)

model_paper.load_data()
fit_paper = model_paper.fit(CYCLES=1, WARMUP=1000, SAMPLES=1500)
```

### Scenario 2: Sensor Placement Analysis

```python
# Use the FloodNet analysis notebook
# notebooks/for_floodnet/f_floodnet_placement.ipynb
```

### Scenario 3: Bias Analysis

```python
# Use the bias analysis notebook
# notebooks/for_paper/f_311_biases.ipynb
```

## Troubleshooting Quick Start

### Issue 1: Data Not Found

**Error**: `FileNotFoundError: data/processed/flooding_ct_dataset.csv`

**Solution**:
```python
# Check if data exists
import os
print("Available files in data/processed/:")
print(os.listdir("data/processed/"))

# If no data, you may need to:
# 1. Download the data
# 2. Process raw data first
# 3. Use simulated data for testing
```

### Issue 2: Stan Model Compilation

**Error**: `Stan model compilation failed`

**Solution**:
```python
# Check Stan installation
import cmdstanpy
print(f"Stan version: {cmdstanpy.__version__}")

# Try recompiling
model = ICAR_MODEL(...)
model.load_data()
# The model will recompile automatically
```

### Issue 3: Memory Issues

**Error**: `MemoryError`

**Solution**:
```python
# Use smaller sample sizes
fit = model.fit(CYCLES=1, WARMUP=500, SAMPLES=500)

# Or use data downsampling
model = ICAR_MODEL(..., downsample_frac=0.1)
```

## Next Steps

After completing the quick start:

1. **Explore the notebooks**: Check out the analysis notebooks in `notebooks/`
2. **Read the full documentation**: See `docs/` for detailed guides
3. **Customize the analysis**: Modify parameters for your specific needs
4. **Contribute**: Submit issues or pull requests

## Getting Help

- **Documentation**: Check the `docs/` folder
- **Issues**: Use GitHub issues for bugs
- **Discussions**: Use GitHub discussions for questions
- **Email**: Contact the development team

## Performance Tips

### For Quick Testing
- Use `downsample_frac=0.1` for faster runs
- Reduce `WARMUP` and `SAMPLES` parameters
- Use `SIMULATED_DATA=True` for testing

### For Production Runs
- Use full dataset with `downsample_frac=1.0`
- Increase `WARMUP` to 2000+ and `SAMPLES` to 3000+
- Enable external covariates for better results
- Use multiple chains for robust inference 