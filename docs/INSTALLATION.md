# Installation Guide

This guide provides detailed instructions for setting up the BayFlood environment.

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows (Linux recommended)
- **Python**: 3.8 or higher (3.10 recommended)
- **Memory**: At least 8GB RAM (16GB+ recommended for large datasets)
- **Storage**: At least 10GB free space
- **Internet**: Required for downloading dependencies and data

### Required Software

1. **Python**: Download from [python.org](https://python.org)
2. **Conda** (recommended): Download from [conda.io](https://conda.io)
3. **Git**: For version control
4. **Stan**: For Bayesian modeling

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd bayflood
```

### 2. Set Up Python Environment

#### Option A: Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n bayflood python=3.10 -y

# Activate the environment
conda activate bayflood

# Install core dependencies via conda
conda install -c conda-forge numpy scipy pandas scikit-learn matplotlib seaborn jupyter -y
conda install -c conda-forge geopandas shapely pyproj fiona rasterio -y
conda install -c conda-forge pyarrow h3 osmnx geopy -y
```

#### Option B: Using Virtual Environment

```bash
# Create a virtual environment
python -m venv bayflood_env

# Activate the environment
# On Linux/macOS:
source bayflood_env/bin/activate
# On Windows:
bayflood_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Python Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt
```

### 4. Install Stan

Stan is required for Bayesian modeling. Install using one of these methods:

#### Option A: Using CmdStanPy (Recommended)

```bash
pip install cmdstanpy
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

#### Option B: Using PyStan

```bash
pip install pystan
```

#### Option C: Manual Installation

1. Download Stan from [mc-stan.org](https://mc-stan.org)
2. Follow platform-specific installation instructions
3. Set environment variables if needed

### 5. Verify Installation

Run the verification script:

```bash
python -c "
import numpy as np
import pandas as pd
import geopandas as gpd
import stan
import matplotlib.pyplot as plt
print('✅ All core dependencies installed successfully!')
"
```

## Environment Configuration

### 1. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Create .env file
touch .env
```

Add the following variables to `.env`:

```env
# Stan configuration
STAN_BACKEND=cmdstanpy
STAN_THREADS=4

# Data paths (adjust as needed)
DATA_DIR=./data
PROCESSED_DATA_DIR=./data/processed
AGGREGATION_DIR=./aggregation

# Logging
LOG_LEVEL=INFO
LOG_FILE=./log/analysis.log

# Model parameters
DEFAULT_WARMUP=1000
DEFAULT_SAMPLES=1500
```

### 2. Create Required Directories

```bash
# Create directory structure
mkdir -p data/processed
mkdir -p data/raw
mkdir -p aggregation/flooding
mkdir -p aggregation/demo
mkdir -p aggregation/geo
mkdir -p runs
mkdir -p log
mkdir -p deliverables
```

## Data Setup

### 1. Download Required Data

The analysis requires several data sources. Create a `data/` directory structure:

```
data/
├── raw/                    # Raw data files
├── processed/              # Processed datasets
└── external/               # External data sources
```

### 2. Data Sources

You'll need to obtain the following data:

1. **Census Tract Boundaries**: NYC census tract GeoJSON files
2. **Demographic Data**: ACS 2023 data for NYC
3. **311 Complaint Data**: NYC 311 flooding complaints
4. **FloodNet Sensor Data**: Sensor locations and readings
5. **Topographic Data**: Elevation and terrain data
6. **Dashcam Imagery**: Processed flood detection results

### 3. Data Processing

Run the data processing scripts:

```bash
# Process raw data (if needed)
python scripts/process_data.py

# Validate data integrity
python scripts/validate_data.py
```

## Troubleshooting

### Common Issues

#### 1. Stan Installation Problems

**Error**: `Stan installation not found`

**Solution**:
```bash
# Reinstall Stan
pip uninstall cmdstanpy pystan
pip install cmdstanpy
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

#### 2. Geospatial Library Issues

**Error**: `GDAL not found`

**Solution**:
```bash
# Install GDAL via conda
conda install -c conda-forge gdal

# Or install system dependencies
# Ubuntu/Debian:
sudo apt-get install libgdal-dev
# macOS:
brew install gdal
```

#### 3. Memory Issues

**Error**: `MemoryError` during model fitting

**Solution**:
- Reduce sample size in model parameters
- Use data downsampling
- Increase system memory or use cloud computing

#### 4. Python Version Issues

**Error**: `SyntaxError` or import errors

**Solution**:
```bash
# Check Python version
python --version

# Ensure Python 3.8+ is installed
conda install python=3.10
```

### Platform-Specific Notes

#### Linux

- Install system dependencies: `sudo apt-get install libgdal-dev libproj-dev`
- Use conda for easier dependency management

#### macOS

- Install Xcode command line tools: `xcode-select --install`
- Use Homebrew for system dependencies: `brew install gdal proj`

#### Windows

- Use WSL2 for better compatibility
- Install Visual Studio Build Tools for C++ compilation
- Use conda instead of pip for complex packages

## Performance Optimization

### 1. Parallel Processing

Set the number of CPU cores for parallel processing:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### 2. Stan Configuration

Optimize Stan for your system:

```python
# In your Python code
import cmdstanpy
cmdstanpy.set_cmdstan_path('/path/to/cmdstan')
```

### 3. Memory Management

For large datasets, consider:
- Using data chunking
- Implementing memory-efficient data loading
- Using cloud computing resources

## Next Steps

After successful installation:

1. **Read the Quick Start Guide**: See `docs/QUICKSTART.md`
2. **Prepare your data**: Follow the Data Preparation Guide
3. **Run a test analysis**: Use the example notebooks
4. **Explore the codebase**: Review the main modules

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the GitHub issues page
3. Create a new issue with detailed error information
4. Contact the development team

## Version Compatibility

This installation guide is compatible with:
- Python: 3.8-3.11
- Stan: 2.32+
- Operating Systems: Linux, macOS, Windows 