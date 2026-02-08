import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

def get_bias_metrics(model, var):
    """
    Extract bias metrics from a fitted statsmodels model for a given variable.
    
    Parameters:
    -----------
    model : statsmodels.discrete.discrete_model.BinaryResultsWrapper
        Fitted logistic regression model
    var : str
        Name of the variable to extract metrics for
        
    Returns:
    --------
    pd.DataFrame
        Single row dataframe with metrics for the variable
    """
    params = model.params
    conf_int = model.conf_int()
    
    return pd.DataFrame({
        'variable': [var],
        'mean': [params[var]],
        'ci_lower': [conf_int.loc[var, 0]],
        'ci_upper': [conf_int.loc[var, 1]],
        'p_value': [model.pvalues[var]],
        'std_err': [model.bse[var]]
    })

def analyze_biases(data, outcome='any_sensors', predictor='p_y', 
                  variables=['white_frac', 'black_frac', 'hispanic_frac', 
                           'asian_frac', 'median_household_income']):
    """
    Analyze biases across multiple variables using logistic regression.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    outcome : str
        Name of binary outcome variable
    predictor : str
        Name of main predictor variable
    variables : list
        List of variables to analyze for bias
        
    Returns:
    --------
    pd.DataFrame
        Compiled bias metrics for all variables
    """
    bias_df = pd.DataFrame()
    
    for var in variables:
        # Standardize continuous variables
        data_std = data.copy()
        data_std[var] = (data_std[var] - data_std[var].mean()) / data_std[var].std()
        
        # Fit model with interaction
        formula = f'{outcome} ~ {predictor} + {var}'  # Simplified model without interaction
        try:
            model = sm.Logit.from_formula(formula, data=data_std).fit(
                method='bfgs',
                cov_type='HC0'  # Use robust standard errors
            )
            bias_metrics = get_bias_metrics(model, var)
            bias_df = pd.concat([bias_df, bias_metrics], ignore_index=True)
        except Exception as e:
            print(f"Error fitting model for {var}: {str(e)}")
            
    return bias_df

# The plotting function remains the same
def plot_bias_results(bias_df, title="Bias Analysis Results"):
    """
    Create a forest plot of bias analysis results.
    
    Parameters:
    -----------
    bias_df : pd.DataFrame
        Output from analyze_biases function
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Create forest plot
    y_pos = np.arange(len(bias_df))
    
    plt.errorbar(x=bias_df['mean'], y=y_pos,
                xerr=np.array([bias_df['mean'] - bias_df['ci_lower'], 
                              bias_df['ci_upper'] - bias_df['mean']]),
                fmt='o', capsize=5)
    
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.yticks(y_pos, bias_df['variable'])
    
    plt.xlabel('Coefficient Estimate')
    plt.title(title)
    plt.tight_layout()
    
    return plt