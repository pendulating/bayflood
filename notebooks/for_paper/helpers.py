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

def add_demo_cols(analysis_df: pd.DataFrame):
    analysis_df['frac_white'] = analysis_df['nhl_white_alone'] / analysis_df['total_population'] 
    analysis_df['frac_black'] = analysis_df['nhl_black_alone'] / analysis_df['total_population']
    analysis_df['frac_hispanic'] = analysis_df['hispanic_alone'] / analysis_df['total_population']
    analysis_df['frac_asian'] = analysis_df['nhl_asian_alone'] / analysis_df['total_population']

    # educational attainment 
    analysis_df['frac_hs'] = analysis_df['num_high_school_graduates'] / analysis_df['total_population']
    analysis_df['frac_bachelors'] = analysis_df['num_bachelors_degree'] / analysis_df['total_population']
    analysis_df['frac_grad'] = analysis_df['num_graduate_degree'] / analysis_df['total_population']


    # age 
    analysis_df['frac_children'] = analysis_df['n_children'] / analysis_df['total_population']
    analysis_df['frac_elderly']  = analysis_df['n_elderly'] / analysis_df['total_population']

    # internet 
    analysis_df['frac_internet'] = analysis_df['num_households_with_internet'] / analysis_df['total_households']
    analysis_df['frac_smartphone'] = analysis_df['num_households_with_smartphone'] / analysis_df['total_households']

    # english first language 
    analysis_df['frac_limited_english'] = analysis_df['num_limited_english_speaking_households'] / analysis_df['total_households']

    return analysis_df



def latex(flag: bool):
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