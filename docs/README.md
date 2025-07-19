# Documentation Index

Welcome to the BayFlood documentation. This guide provides an overview of all available documentation and helps you find the information you need.

## Quick Navigation

### 🚀 Getting Started
- **[Installation Guide](INSTALLATION.md)** - Complete setup instructions
- **[Quick Start Guide](QUICKSTART.md)** - Get up and running in 30 minutes
- **[User Guide](USER_GUIDE.md)** - Comprehensive analysis instructions

### 📚 Reference Documentation
- **[API Reference](API_REFERENCE.md)** - Detailed function and class documentation
- **[Model Specifications](MODEL_SPECS.md)** - Stan model details and theory
- **[Data Format Guide](DATA_FORMAT.md)** - Data requirements and formats

### 🔧 Advanced Topics
- **[Advanced Analysis](ADVANCED_ANALYSIS.md)** - Custom analyses and extensions
- **[Performance Optimization](PERFORMANCE.md)** - Speed and memory optimization
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions

## Documentation Structure

```
docs/
├── README.md                    # This file - documentation index
├── INSTALLATION.md              # Environment setup
├── QUICKSTART.md                # Quick start guide
├── USER_GUIDE.md                # Comprehensive user guide
├── API_REFERENCE.md             # API documentation
├── MODEL_SPECS.md               # Model specifications
├── DATA_FORMAT.md               # Data format requirements
├── ADVANCED_ANALYSIS.md         # Advanced analysis techniques
├── PERFORMANCE.md               # Performance optimization
├── TROUBLESHOOTING.md           # Troubleshooting guide
└── examples/                    # Example scripts and notebooks
    ├── basic_analysis.py
    ├── custom_model.py
    └── visualization_examples.py
```

## Getting Started

### For New Users

1. **Start with Installation**: Follow the [Installation Guide](INSTALLATION.md) to set up your environment
2. **Quick Start**: Use the [Quick Start Guide](QUICKSTART.md) to run your first analysis
3. **Learn the Basics**: Read the [User Guide](USER_GUIDE.md) for comprehensive instructions

### For Experienced Users

1. **Skip to User Guide**: Go directly to [User Guide](USER_GUIDE.md) for detailed analysis instructions
2. **Check API Reference**: Use [API Reference](API_REFERENCE.md) for function documentation
3. **Advanced Topics**: Explore [Advanced Analysis](ADVANCED_ANALYSIS.md) for custom techniques

### For Developers

1. **Model Specifications**: Review [Model Specifications](MODEL_SPECS.md) for Stan model details
2. **API Reference**: Use [API Reference](API_REFERENCE.md) for development
3. **Performance**: Check [Performance Optimization](PERFORMANCE.md) for optimization tips

## Key Concepts

### ICAR Model
The Intrinsic Conditional Autoregressive (ICAR) model is the core statistical method used for spatial analysis of flooding patterns in BayFlood. It accounts for geographic dependencies between census tracts.

### Data Sources
The analysis integrates multiple data sources:
- **Dashcam imagery**: Automated flood detection from vehicle cameras
- **311 complaints**: Citizen-reported flooding incidents
- **FloodNet sensors**: Physical flood monitoring sensors
- **Census data**: Demographic and socioeconomic information
- **Topographic data**: Elevation and terrain characteristics

### Spatial Analysis
The framework uses census tracts as the primary spatial unit and incorporates:
- Spatial adjacency matrices
- Geographic covariates
- Spatial autocorrelation modeling

## Common Use Cases

### 1. Reproduce Paper Results
Follow the [User Guide](USER_GUIDE.md) to reproduce the main findings from the BayFlood research paper.

### 2. Custom Analysis
Use the [Advanced Analysis](ADVANCED_ANALYSIS.md) guide to conduct your own BayFlood research.

### 3. Data Integration
Reference the [Data Format Guide](DATA_FORMAT.md) to integrate your own data sources with BayFlood.

### 4. Model Customization
Check [Model Specifications](MODEL_SPECS.md) to understand and modify the BayFlood Stan models.

## Code Examples

### Basic Analysis
```python
from icar_model import ICAR_MODEL

model = ICAR_MODEL(
    PREFIX='my_analysis',
    ICAR_PRIOR_SETTING="icar",
    ANNOTATIONS_HAVE_LOCATIONS=True,
    EXTERNAL_COVARIATES=True,
    ESTIMATE_PARAMS=['p_y', 'at_least_one_positive_image_by_area'],
    EMPIRICAL_DATA_PATH="data/processed/flooding_ct_dataset.csv"
)

model.load_data()
fit = model.fit(CYCLES=1, WARMUP=1000, SAMPLES=1500)
```

### Generate Maps
```python
from generate_maps import generate_maps

generate_maps(
    run_id='my_analysis',
    estimate_path='runs/my_analysis/estimate_at_least_one_positive_image_by_area.csv',
    estimate='at_least_one_positive_image_by_area'
)
```

### Comprehensive Analysis
```python
from analysis_df import generate_nyc_analysis_df

analysis_df = generate_nyc_analysis_df(
    run_dir='runs/my_analysis',
    custom_prefix='comprehensive',
    use_smoothing=True
)
```

## Troubleshooting

### Common Issues

1. **Installation Problems**: See [Installation Guide](INSTALLATION.md) troubleshooting section
2. **Data Loading Errors**: Check [Data Format Guide](DATA_FORMAT.md)
3. **Model Convergence**: Review [Troubleshooting Guide](TROUBLESHOOTING.md)
4. **Performance Issues**: See [Performance Optimization](PERFORMANCE.md)

### Getting Help

- **Documentation**: Check the relevant guide in this documentation
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share solutions
- **Email**: Contact the development team for specific issues

## Contributing to Documentation

### Reporting Issues
If you find errors or gaps in the documentation:

1. Check if the issue is already reported
2. Create a new GitHub issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs. actual behavior
   - Relevant file/page references

### Suggesting Improvements
To suggest documentation improvements:

1. Create a GitHub issue with your suggestion
2. Provide specific examples or use cases
3. Explain the benefit of the proposed change

### Contributing Content
To contribute documentation:

1. Fork the repository
2. Create a new branch for your changes
3. Write clear, well-structured documentation
4. Include examples and code snippets
5. Submit a pull request

## Documentation Standards

### Writing Style
- Use clear, concise language
- Include practical examples
- Provide step-by-step instructions
- Use consistent formatting

### Code Examples
- Include complete, runnable examples
- Add comments explaining key steps
- Use realistic data and parameters
- Test all examples before publishing

### Structure
- Use clear headings and subheadings
- Include table of contents for long documents
- Cross-reference related sections
- Maintain consistent file naming

## Version Information

This documentation corresponds to:
- **Repository Version**: 1.0.0
- **Python Version**: 3.8+
- **Stan Version**: 2.32+
- **Last Updated**: [Current Date]

## License

This documentation is provided under the same license as the main repository. See the main README.md for license details.

## Acknowledgments

Thanks to all contributors who have helped improve this documentation. Special thanks to the research team and community members who provided feedback and suggestions. 