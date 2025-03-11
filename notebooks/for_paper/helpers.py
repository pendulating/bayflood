import pandas as pd
from logger import setup_logger 
log = setup_logger("analysis-helpers")
log.setLevel("INFO")

def add_covariate_cols(analysis_df: pd.DataFrame):
    analysis_df['any_sensors'] = analysis_df['n_floodnet_sensors'] > 0
    log.info(f"Found {analysis_df['any_sensors'].sum()} tracts with at least one FloodNet sensor.")

    # get all columns with 311 in name, and sum them up
    analysis_df['n_311_reports'] = analysis_df.filter(like='311').sum(axis=1)
    log.info(f"Found {analysis_df['n_311_reports'].sum()} 311 requests.")

    analysis_df['any_311_report'] = analysis_df['n_311_reports'] > 0
    log.info(f"Found {analysis_df['any_311_report'].sum()} tracts with at least one 311 report.")

    analysis_df['no_dep_flooding'] = analysis_df['dep_moderate_2_frac'] == 0
    log.info(f"Found {analysis_df['no_dep_flooding'].sum()} tracts with no DEP flooding.")

    # Check for NA values instead of asserting
    covariate_cols = ['any_sensors', 'n_311_reports', 'any_311_report', 'no_dep_flooding']
    na_counts = analysis_df[covariate_cols].isna().sum()
    total_na = na_counts.sum()
    if total_na > 0:
        log.warning(f"Found {total_na} NA values in covariate columns.")
        for col, count in na_counts.items():
            if count > 0:
                log.warning(f"Column '{col}' has {count} NA values.")

    return analysis_df

def add_estimate_cols(analysis_df: pd.DataFrame):
    analysis_df['confirmed_flooding'] = (analysis_df['at_least_one_positive_image_by_area'] == 1)
    log.info(f"Found {analysis_df['confirmed_flooding'].sum()} tracts with confirmed flooding.")

    analysis_df['above_thres'] = analysis_df['p_y'] > analysis_df[analysis_df['confirmed_flooding']]['p_y'].quantile(0.25)
    log.info(f"Found {analysis_df['above_thres'].sum()} tracts with above threshold ({analysis_df[analysis_df['confirmed_flooding']]['p_y'].quantile(0.25)}) flooding.")

    analysis_df['confirmed_or_above_thres'] = analysis_df['confirmed_flooding'] | analysis_df['above_thres']
    log.info(f"Found {analysis_df['confirmed_or_above_thres'].sum()} tracts with confirmed or above threshold flooding.")

    # Check for NA values instead of asserting
    estimate_cols = ['confirmed_flooding', 'above_thres', 'confirmed_or_above_thres']
    na_counts = analysis_df[estimate_cols].isna().sum()
    total_na = na_counts.sum()
    if total_na > 0:
        log.warning(f"Found {total_na} NA values in estimate columns.")
        for col, count in na_counts.items():
            if count > 0:
                log.warning(f"Column '{col}' has {count} NA values.")

    return analysis_df

def add_demo_cols(analysis_df: pd.DataFrame):
    analysis_df['frac_white'] = analysis_df['nhl_white_alone'] / analysis_df['total_population'] 
    log.info("Added fraction white (frac_white) column.")

    analysis_df['frac_black'] = analysis_df['nhl_black_alone'] / analysis_df['total_population']
    log.info("Added fraction black (frac_black) column.")

    analysis_df['frac_hispanic'] = analysis_df['hispanic_alone'] / analysis_df['total_population']
    log.info("Added fraction hispanic (frac_hispanic) column.")

    analysis_df['frac_asian'] = analysis_df['nhl_asian_alone'] / analysis_df['total_population']
    log.info("Added fraction asian (frac_asian) column.")

    # educational attainment 
    analysis_df['frac_hs'] = analysis_df['num_high_school_graduates'] / analysis_df['total_population']
    log.info("Added fraction high school graduates (frac_hs) column.")

    analysis_df['frac_bachelors'] = analysis_df['num_bachelors_degree'] / analysis_df['total_population']
    log.info("Added fraction bachelors degree (frac_bachelors) column.")

    analysis_df['frac_grad'] = analysis_df['num_graduate_degree'] / analysis_df['total_population']
    log.info("Added fraction graduate degree (frac_grad) column.")


    # age 
    analysis_df['frac_children'] = analysis_df['n_children'] / analysis_df['total_population']
    log.info("Added fraction children (frac_children) column.")

    analysis_df['frac_elderly']  = analysis_df['n_elderly'] / analysis_df['total_population']
    log.info("Added fraction elderly (frac_elderly) column.")

    # internet 
    analysis_df['frac_internet'] = analysis_df['num_households_with_internet'] / analysis_df['total_households']
    log.info("Added fraction internet (frac_internet) column.")

    analysis_df['frac_smartphone'] = analysis_df['num_households_with_smartphone'] / analysis_df['total_households']
    log.info("Added fraction smartphone (frac_smartphone) column.")

    # english first language 
    analysis_df['frac_limited_english'] = analysis_df['num_limited_english_speaking_households'] / analysis_df['total_households']
    log.info("Added fraction limited english speaking households (frac_limited_english) column.")

    # Check for NA values instead of asserting
    demo_cols = ['frac_white', 'frac_black', 'frac_hispanic', 'frac_asian', 'frac_hs', 'frac_bachelors', 
                'frac_grad', 'frac_children', 'frac_elderly', 'frac_internet', 'frac_smartphone', 
                'frac_limited_english']
    na_counts = analysis_df[demo_cols].isna().sum()
    total_na = na_counts.sum()
    if total_na > 0:
        log.warning(f"Found {total_na} NA values in demographic columns.")
        for col, count in na_counts.items():
            if count > 0:
                log.warning(f"Column '{col}' has {count} NA values.")

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