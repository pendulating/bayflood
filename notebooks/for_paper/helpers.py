import pandas as pd
from logger import setup_logger 
log = setup_logger("analysis-helpers")
log.setLevel("INFO")

def add_helper_cols(analysis_df: pd.DataFrame):
    analysis_df['any_sensors'] = analysis_df['n_floodnet_sensors'] > 0
    log.info(f"Found {analysis_df['any_sensors'].sum()} tracts with at least one FloodNet sensor.")

    # get all columns with 311 in name, and sum them up
    analysis_df['n_311_reports'] = analysis_df.filter(like='311').sum(axis=1)
    log.info(f"Found {analysis_df['n_311_reports'].sum()} 311 requests.")

    analysis_df['any_311_report'] = analysis_df['n_311_reports'] > 0
    log.info(f"Found {analysis_df['any_311_report'].sum()} tracts with at least one 311 report.")

    analysis_df['no_dep_flooding'] = analysis_df['dep_moderate_2_frac'] == 0
    log.info(f"Found {analysis_df['no_dep_flooding'].sum()} tracts with no DEP flooding.")

    return analysis_df


def enable_latex(flag: bool):
    if flag:
        log.info("Enabling LaTeX for matplotlib.")
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
    else:
        log.info("Disabling LaTeX for matplotlib.")
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=False)
        plt.rc('font', family='serif')