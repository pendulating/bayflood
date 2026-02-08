"""
Utility functions for bayflood pipeline data processing.

This module provides functions for reading and processing flooding data,
generating simulated data, and validating observed data for the ICAR model.
"""

import pandas as pd
import numpy as np
from scipy.special import expit
import random

TOTAL_ANNOTATED_CLASSIFIED_NEGATIVE = 500
TOTAL_ANNOTATED_CLASSIFIED_POSITIVE = 500
TOTAL_ANNOTATED_CLASSIFIED_NEGATIVE_TRUE_POSITIVE = 3
TOTAL_ANNOTATED_CLASSIFIED_POSITIVE_TRUE_POSITIVE = 329


from logger import setup_logger 
logger = setup_logger("util-subroutine")
logger.setLevel("INFO")


def read_real_data(
    fpath: str,
    annotations_have_locations=False,
    adj=[],
    adj_matrix_storage=False,
    use_external_covariates=False,
    use_catch_basins=True,
    id_column="GEOID"
):
    """
    Read and process real flooding data for ICAR model analysis.
    
    Loads census geography-level flooding data and prepares it for Stan model fitting.
    Handles adjacency matrix construction, external covariates processing, and
    data validation. Works with any census geography level (tracts, block groups, blocks).
    
    Parameters:
    -----------
    fpath : str
        Path to the CSV file containing flooding data (required)
    annotations_have_locations : bool, default=False
        Whether the dataset includes annotation location information
    adj : list, default=[]
        List of file paths for adjacency information. If adj_matrix_storage=True,
        should contain one path to .npy file. Otherwise, should contain two paths
        to .txt files with node1 and node2 lists.
    adj_matrix_storage : bool, default=False
        Whether adjacency data is stored as matrix (.npy) or edge lists (.txt)
    use_external_covariates : bool, default=False
        Whether to include external covariates in the model
    use_catch_basins : bool, default=True
        Whether to include catch basin covariates (n_catch_basins, catch_basin_density, cb_days_clogged,
        cb_avg_resolution_time, has_clogged_cb_complaint) when use_external_covariates is True
    id_column : str, default="GEOID"
        Name of the column containing the geography identifier (e.g., "GEOID", "GEOID20")
        
    Returns:
    --------
    tuple
        - observed_data (dict): Dictionary containing data for Stan model
        - external_covariates_info (dict): Information about external covariates
          (empty dict if use_external_covariates=False)
        
    Raises:
    ------
    FileNotFoundError
        If the data file or adjacency files are not found
    ValueError
        If data format is invalid or required columns are missing
        
    Example:
    --------
    >>> observed_data, cov_info = read_real_data(
    ...     fpath="data/processed/flooding_ct_dataset.csv",  # or flooding_cbg_dataset.csv
    ...     annotations_have_locations=True,
    ...     use_external_covariates=True,
    ...     id_column="GEOID"  # use GEOID20 for census blocks
    ... )
    """
    single_compartment_for_debugging = False
    df = pd.read_csv(fpath)

    df[["total_images", "positive_images"]] = (
        df[["n_total", "n_classified_positive"]].astype(int).fillna(0)
    )
    N = len(df)
    n_images_by_area = df["total_images"].values
    n_classified_positive_by_area = df["positive_images"].values
    
    # Store geoid (the ID column of df) as an array of stan real numerics
    # For backward compatibility, also check for GEOID if id_column not found
    if id_column not in df.columns and "GEOID" in df.columns:
        id_column = "GEOID"
        logger.warning(f"id_column not found, falling back to 'GEOID'")
    geoid = df[id_column].astype(float).values


    if len(adj) > 0:
        if adj_matrix_storage: 
            # read adjm from .npy 
            adjm = np.load(adj[0])
            logger.info(f"Read adjm of shape: {adjm.shape}")
        else:
            # make sure there are two elements in the adj arg 
            assert len(adj) == 2
            # read node1 from .txt in adj[0]
            with open(adj[0], "r") as f:
                node1 = [int(line) for line in f]
            # read node2 from .txt in adj[1]
            with open(adj[1], "r") as f:
                node2 = [int(line) for line in f]

            # Pass 1: Ensure node1 < node2 in every row
            ordered_node1 = []
            ordered_node2 = []
            for i in range(len(node1)):
                assert node1[i] != node2[i]  # No self-loops
                if node1[i] < node2[i]:
                    ordered_node1.append(node1[i])
                    ordered_node2.append(node2[i])
                else:
                    ordered_node1.append(node2[i])
                    ordered_node2.append(node1[i])

            # Pass 2: Drop duplicates while preserving order
            seen_edges = set()
            unique_node1 = []
            unique_node2 = []
            for i in range(len(ordered_node1)):
                edge = (ordered_node1[i], ordered_node2[i])
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    unique_node1.append(ordered_node1[i])
                    unique_node2.append(ordered_node2[i])

            node1 = unique_node1
            node2 = unique_node2
            assert (np.array(node1) < np.array(node2)).all()
            

            logger.info(f"Read node1 of length: {len(node1)}")
            logger.info(f"Read node2 of length: {len(node2)}")
            

    else: 

        if adj_matrix_storage: 
            logger.info("No adjm provided, initializing adjm to zeros")
            adjm = np.zeros((N, N))
        else: 
            logger.info("No adj provided, initializing node1 and node2 to empty lists")
            node1 = []
            node2 = []




    if single_compartment_for_debugging:
        N = 1
        n_images_by_area = [sum(n_images_by_area)]
        n_classified_positive_by_area = [sum(n_classified_positive_by_area)]

    if annotations_have_locations:
        df[
            [
                "n_true_positive_classified_positive",
                "n_true_positive_classified_negative",
                "n_true_negative_classified_positive",
                "n_true_negative_classified_negative",
                "n_not_annotated_by_area",
                "n_not_annotated_by_area_classified_positive",
            ]
        ] = (
            df[
                [
                    "n_tp",
                    "n_fn",
                    "n_fp",
                    "n_tn",
                    "total_not_annotated",
                    "positives_not_annotated",
                ]
            ]
            .astype(int)
            .fillna(0)
        )

        n_true_positive_classified_positive_by_area = df[
            "n_true_positive_classified_positive"
        ].values
        n_true_positive_classified_negative_by_area = df[
            "n_true_positive_classified_negative"
        ].values
        n_true_negative_classified_positive_by_area = df[
            "n_true_negative_classified_positive"
        ].values
        n_true_negative_classified_negative_by_area = df[
            "n_true_negative_classified_negative"
        ].values

        n_non_annotated_by_area = df["n_not_annotated_by_area"].values
        n_non_annotated_by_area_classified_positive = df[
            "n_not_annotated_by_area_classified_positive"
        ].values
        observed_data = {
                "N": N,
                "N_edges": len(node1),
                "node1": node1,
                "node2": node2,
                "geoid": geoid,
                # Backward compatibility: also include tract_id pointing to same data
                "tract_id": geoid,
                "n_images_by_area": n_images_by_area,
                "n_classified_positive_by_area": n_classified_positive_by_area,
                "n_classified_positive_annotated_positive_by_area": n_true_positive_classified_positive_by_area,
                "n_classified_positive_annotated_negative_by_area": n_true_negative_classified_positive_by_area,
                "n_classified_negative_annotated_negative_by_area": n_true_negative_classified_negative_by_area,
                "n_classified_negative_annotated_positive_by_area": n_true_positive_classified_negative_by_area,
                "n_non_annotated_by_area": n_non_annotated_by_area,
                "n_non_annotated_by_area_classified_positive": n_non_annotated_by_area_classified_positive, 
                "center_of_phi_offset_prior":-5
            }
        
        # column of ones for external covariates
        observed_data['external_covariates'] = np.ones((N, 1)) # just an intercept term by default. 
        if use_external_covariates:

            def robust_zscore(x, eps=1e-8):
                """Z-score with numeric stability check."""
                mean = np.mean(x)
                std = np.std(x)
                if std < eps:  # Essentially constant feature
                    print(f"Warning: Nearly constant feature detected with std={std}")
                    return np.zeros_like(x)
                return (x - mean) / std

            def check_binary_frequency(x):
                """Check for rare binary events."""
                prop_ones = np.mean(x)
                if prop_ones < 0.01 or prop_ones > 0.99:
                    print(f"Warning: Highly imbalanced binary feature detected with proportion={prop_ones}")


            def process_covariates(df, observed_data, use_external_covariates=True, use_catch_basins=True):
                """Process covariates with robust handling for large N observations."""
                if not use_external_covariates:
                    return {"observed_data": observed_data}


                # 1. Process right-skewed continuous variables
                skewed_cols = [
                    'ft_elevation_min', 
                    'n_311_reports',
                    'n_floodnet_sensors',
                ]
                
                # Conditionally add catch basin covariates to skewed columns
                if use_catch_basins:
                    skewed_cols.append('n_catch_basins')
                    skewed_cols.append('cb_days_clogged')
                    skewed_cols.append('cb_avg_resolution_time')

                available_skewed = []
                for col in skewed_cols:
                    if col not in df.columns:
                        print(f"Warning: Missing covariate column {col}; skipping.")
                        continue
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Check for outliers
                    data = df[col]
                    q1, q3 = np.percentile(data, [25, 75])
                    iqr = q3 - q1
                    outlier_mask = (data < q1 - 3*iqr) | (data > q3 + 3*iqr)
                    if np.any(outlier_mask):
                        print(f"Warning: {np.sum(outlier_mask)} outliers detected in {col}")
                    
                    df[f'{col}_log'] = np.log1p(df[col])
                    available_skewed.append(col)
                
                # 3. Collect covariates
                cols_to_use = [f'{col}_log' for col in available_skewed]

                base_cols = ['ft_elevation_mean', 'dep_moderate_1_frac', 'dep_moderate_2_frac']
                available_base = []
                for col in base_cols:
                    if col not in df.columns:
                        print(f"Warning: Missing covariate column {col}; skipping.")
                        continue
                    available_base.append(col)

                cols_to_use += available_base
                
                # Conditionally add catch basin density and binary indicator
                if use_catch_basins:
                    catch_cols = ['catch_basin_density', 'has_clogged_cb_complaint']
                    for col in catch_cols:
                        if col not in df.columns:
                            print(f"Warning: Missing covariate column {col}; skipping.")
                            continue
                        cols_to_use.append(col)
                
                if len(cols_to_use) == 0:
                    print("Warning: No external covariate columns available; using intercept only.")
                    observed_data['external_covariates'] = np.ones((N, 1))
                    observed_data['n_external_covariates'] = 1
                    return {"observed_data": observed_data}, { "external_covariates": {}}

                # Convert to numeric matrix
                feature_matrix = df[cols_to_use].apply(pd.to_numeric, errors='coerce').values.astype(float)
                feature_matrix_columns = cols_to_use
                print(feature_matrix_columns)
                
                external_covariate_matrix = np.hstack([feature_matrix])
                
                # Z-score everything with robust checks
                n_features = external_covariate_matrix.shape[1]
                for i in range(n_features):
                    col_name = feature_matrix_columns[i]
                    col_data = external_covariate_matrix[:, i].astype(float)
                    
                    # Now we can safely check for NaN/inf
                    if np.any(np.isnan(col_data)) or np.any(np.isinf(col_data)):
                        print(f"Warning: Missing or infinite values detected in {col_name}")
                        col_data = np.nan_to_num(col_data, nan=0, posinf=0, neginf=0)
                    
                    external_covariate_matrix[:, i] = robust_zscore(col_data)
                    print(f"Z-scoring {col_name}: mean={np.mean(col_data):.4f}, std={np.std(col_data):.4f}")
                
                # Update observed data
                if 'external_covariates' in observed_data:
                    observed_data['external_covariates'] = np.hstack((
                        observed_data['external_covariates'], 
                        external_covariate_matrix
                    ))
                    # create dataframe of covariate labels, zscored means, and zscored stds 
                    external_covariates_info = pd.DataFrame({
                        'covariate': feature_matrix_columns,
                        'mean': np.mean(external_covariate_matrix, axis=0),
                        'std': np.std(external_covariate_matrix, axis=0)
                    })
                else:
                    observed_data['external_covariates'] = external_covariate_matrix

                    external_covariates_info = pd.DataFrame({
                        'covariate': feature_matrix_columns,
                        'mean': np.mean(external_covariate_matrix, axis=0),
                        'std': np.std(external_covariate_matrix, axis=0)
                    })
                
                observed_data['n_external_covariates'] = observed_data['external_covariates'].shape[1]
                print(f"Final external covariate matrix shape: {observed_data['external_covariates'].shape}")
                
                return {"observed_data": observed_data}, { "external_covariates": external_covariates_info}
            
            return process_covariates(df, observed_data, use_external_covariates, use_catch_basins)
        else: 
            observed_data['external_covariates'] = np.ones((N, 1)) # just an intercept term by default.
            observed_data['n_external_covariates'] = 1
            return {"observed_data": observed_data}, { "external_covariates": {}}
    else:
        return {
            "observed_data": {
                "N": N,
                "N_edges": len(node1),
                "node1": node1,
                "node2": node2,
                "geoid": geoid,
                # Backward compatibility: also include tract_id pointing to same data
                "tract_id": geoid,
                "n_images_by_area": n_images_by_area,
                "n_classified_positive_by_area": n_classified_positive_by_area,
                "total_annotated_classified_negative": TOTAL_ANNOTATED_CLASSIFIED_NEGATIVE,
                "total_annotated_classified_positive": TOTAL_ANNOTATED_CLASSIFIED_POSITIVE,
                "total_annotated_classified_negative_true_positive": TOTAL_ANNOTATED_CLASSIFIED_NEGATIVE_TRUE_POSITIVE,
                "total_annotated_classified_positive_true_positive": TOTAL_ANNOTATED_CLASSIFIED_POSITIVE_TRUE_POSITIVE,
            }
        }, 
        { "external_covariates": {} }


def validate_observed_data(observed_data, annotations_have_locations=False, downsample_frac=1):
    """
    Validate observed data dictionary for ICAR model fitting.
    
    Parameters
    ----------
    observed_data : dict
        Dictionary containing observed data for the model
    annotations_have_locations : bool, default=False
        Whether annotations include location information
    downsample_frac : float, default=1
        Fraction of data used (validation skipped if != 1)
    """
    # if downsample_frac != 1, return True. binomial sampling will mess up the counts.
    if downsample_frac != 1:
        return True

    # first, logic split on annotations_have_locations
    if annotations_have_locations:
        REQ_COLS = [
            "N",
            "N_edges",
            "node1",
            "node2",
            "n_images_by_area",
            "n_classified_positive_by_area",
            "n_classified_positive_annotated_positive_by_area",
            "n_classified_positive_annotated_negative_by_area",
            "n_classified_negative_annotated_positive_by_area",
            "n_classified_negative_annotated_negative_by_area",
            "n_non_annotated_by_area",
            "n_non_annotated_by_area_classified_positive",
        ]

        DF_COLS = [
            "n_images_by_area",
            "n_classified_positive_by_area",
            "n_classified_positive_annotated_positive_by_area",
            "n_classified_positive_annotated_negative_by_area",
            "n_classified_negative_annotated_positive_by_area",
            "n_classified_negative_annotated_negative_by_area",
            "n_non_annotated_by_area",
            "n_non_annotated_by_area_classified_positive",
        ]

        # make df out of DF_COLS
        df = pd.DataFrame({col: observed_data[col] for col in DF_COLS})
        # write to csv
        df.to_csv("observed_data.csv", index=False)

        for col in REQ_COLS:
            if col not in observed_data:
                raise ValueError(f"Missing required column {col} in observed_data")

        # n_images_by_area - n_not_annotated_by_area should equal the sum of annotated counts
        if not np.allclose(
            np.array(observed_data["n_images_by_area"])
            - np.array(observed_data["n_non_annotated_by_area"]),
            np.array(observed_data["n_classified_positive_annotated_positive_by_area"])
            + np.array(
                observed_data["n_classified_positive_annotated_negative_by_area"]
            )
            + np.array(
                observed_data["n_classified_negative_annotated_positive_by_area"]
            )
            + np.array(
                observed_data["n_classified_negative_annotated_negative_by_area"]
            ),
        ):
            raise ValueError(
                "n_images_by_area - n_not_annotated_by_area should equal the sum of "
                "n_classified_positive_annotated_positive_by_area, "
                "n_classified_positive_annotated_negative_by_area, "
                "n_classified_negative_annotated_positive_by_area, "
                "n_classified_negative_annotated_negative_by_area"
            )

        # sum of n_classified_positive_annotated_positive_by_area and n_classified_positive_annotated_negative_by_area should equal TOTAL_ANNOTATED_CLASSIFIED_POSITIVE
        if (
            sum(observed_data['n_classified_positive_annotated_positive_by_area'])
            + sum(observed_data['n_classified_positive_annotated_negative_by_area'])
            != (TOTAL_ANNOTATED_CLASSIFIED_POSITIVE * downsample_frac)
        ):
            raise ValueError(
                f"sum of n_classified_positive_annotated_positive_by_area ({sum(observed_data['n_classified_positive_annotated_positive_by_area'])}) and n_classified_positive_annotated_negative_by_area ({sum(observed_data['n_classified_positive_annotated_negative_by_area'])}) should equal total_annotated_classified_positive ({(TOTAL_ANNOTATED_CLASSIFIED_POSITIVE * downsample_frac)})"
            )

        # sum of n_classified_negative_annotated_positive_by_area and n_classified_negative_annotated_negative_by_area should equal TOTAL_ANNOTATED_CLASSIFIED_NEGATIVE
        if (
            sum(observed_data['n_classified_negative_annotated_positive_by_area'])
            + sum(observed_data['n_classified_negative_annotated_negative_by_area'])
            != (TOTAL_ANNOTATED_CLASSIFIED_NEGATIVE * downsample_frac)
        ):
            raise ValueError(
                f"sum of n_classified_negative_annotated_positive_by_area ({sum(observed_data['n_classified_negative_annotated_positive_by_area'])})  and n_classified_negative_annotated_negative_by_area ({sum(observed_data['n_classified_negative_annotated_negative_by_area'])}) should equal total_annotated_classified_negative ({(TOTAL_ANNOTATED_CLASSIFIED_NEGATIVE * downsample_frac)})"
            )

    else:
        REQ_COLS = [
            "N",
            "N_edges",
            "node1",
            "node2",
            "n_images_by_area",
            "n_classified_positive_by_area",
            "total_annotated_classified_negative",
            "total_annotated_classified_positive",
            "total_annotated_classified_negative_true_positive",
            "total_annotated_classified_positive_true_positive",
        ]

        for col in REQ_COLS:
            if col not in observed_data:
                raise ValueError(f"Missing required column {col} in observed_data")


def generate_simulated_data(
    N,
    images_per_location,
    total_annotated_classified_negative,
    total_annotated_classified_positive,
    icar_prior_setting,
    annotations_have_locations,
):
    """
    Generate simulated data for the model.
    
    Parameters
    ----------
    N : int
        Number of geographic areas
    images_per_location : int
        Expected number of images per area (Poisson parameter)
    total_annotated_classified_negative : int
        Total number of annotated negative classifications
    total_annotated_classified_positive : int
        Total number of annotated positive classifications
    icar_prior_setting : str
        ICAR prior setting ("none" or other)
    annotations_have_locations : bool
        Whether annotations include location information
        
    Returns
    -------
    dict
        Dictionary containing observed_data and parameters
    """
    node1 = []
    node2 = []
    for i in range(N):
        for j in range(i + 1, N):
            if np.random.rand() < 0.1:
                node1.append(i + 1)  # one indexing for Stan.
                node2.append(j + 1)
    phi_offset = random.random() * -3 - 1  # mean of phi.

    # these only matter for CAR model. https://mc-stan.org/users/documentation/case-studies/mbjoseph-CARStan.html
    D = np.zeros((N, N))
    W = np.zeros((N, N))
    for i in range(len(node1)):
        D[node1[i] - 1, node1[i] - 1] += 1
        D[node2[i] - 1, node2[i] - 1] += 1
        W[node1[i] - 1, node2[i] - 1] = 1
        W[node2[i] - 1, node1[i] - 1] = 1
    B = np.linalg.inv(D) @ W
    tau = np.random.gamma(scale=0.2, shape=2)
    alpha = np.random.random()
    sigma = np.linalg.inv(tau * D @ (np.eye(N) - alpha * B))
    if icar_prior_setting != "none":
        phi = np.random.multivariate_normal(mean=np.zeros(N), cov=sigma)
    else:
        phi = np.random.normal(
            loc=0, size=N
        )  # this uses no icar prior, just draws everything independently.
    p_Y = expit(phi + phi_offset)
    n_images_by_area = np.random.poisson(images_per_location, N)
    p_y_hat_1_given_y_1 = random.random() * 0.5 + 0.2
    p_y_hat_1_given_y_0 = random.random() * 0.01 + 0.01

    n_true_positive_by_area = []
    n_true_positive_classified_positive_by_area = []
    n_true_positive_classified_negative_by_area = []
    n_true_negative_classified_positive_by_area = []
    n_true_negative_classified_negative_by_area = []
    n_classified_positive_by_area = []

    for i in range(N):
        n_true_positive_by_area.append(np.random.binomial(n_images_by_area[i], p_Y[i]))
        n_true_positive_classified_positive_by_area.append(
            np.random.binomial(n_true_positive_by_area[i], p_y_hat_1_given_y_1)
        )
        n_true_positive_classified_negative_by_area.append(
            n_true_positive_by_area[i] - n_true_positive_classified_positive_by_area[i]
        )
        n_true_negative_classified_positive_by_area.append(
            np.random.binomial(
                n_images_by_area[i] - n_true_positive_by_area[i], p_y_hat_1_given_y_0
            )
        )
        n_true_negative_classified_negative_by_area.append(
            n_images_by_area[i]
            - n_true_positive_by_area[i]
            - n_true_negative_classified_positive_by_area[i]
        )
        n_classified_positive_by_area.append(
            n_true_positive_classified_positive_by_area[i]
            + n_true_negative_classified_positive_by_area[i]
        )
    empirical_p_yhat = sum(n_classified_positive_by_area) * 1.0 / sum(n_images_by_area)
    empirical_p_y = sum(n_true_positive_by_area) * 1.0 / sum(n_images_by_area)
    p_y_1_given_y_hat_1 = p_y_hat_1_given_y_1 * empirical_p_y / empirical_p_yhat
    p_y_1_given_y_hat_0 = (
        (1 - p_y_hat_1_given_y_1) * empirical_p_y / (1 - empirical_p_yhat)
    )
    print("empirical_p_y", empirical_p_y)
    print("empirical_p_yhat", empirical_p_yhat)
    print("p_y_hat_1_given_y_1", p_y_hat_1_given_y_1)
    print("p_y_hat_1_given_y_0", p_y_hat_1_given_y_0)
    print("p_y_1_given_y_hat_1", p_y_1_given_y_hat_1)
    print("p_y_1_given_y_hat_0", p_y_1_given_y_hat_0)

    if not annotations_have_locations:
        total_annotated_classified_negative_true_positive = np.random.binomial(
            total_annotated_classified_negative, p_y_1_given_y_hat_0
        )
        total_annotated_classified_positive_true_positive = np.random.binomial(
            total_annotated_classified_positive, p_y_1_given_y_hat_1
        )
        print(
            "number of annotated classified negative which were positive: %i/%i"
            % (
                total_annotated_classified_negative_true_positive,
                total_annotated_classified_negative,
            )
        )
        print(
            "number of annotated classified positive which were positive: %i/%i"
            % (
                total_annotated_classified_positive_true_positive,
                total_annotated_classified_positive,
            )
        )
        observed_data = {
            "N": N,
            "N_edges": len(node1),
            "node1": node1,
            "node2": node2,
            "n_images_by_area": n_images_by_area,
            "n_classified_positive_by_area": n_classified_positive_by_area,
            "total_annotated_classified_negative": total_annotated_classified_negative,
            "total_annotated_classified_positive": total_annotated_classified_positive,
            "total_annotated_classified_negative_true_positive": total_annotated_classified_negative_true_positive,
            "total_annotated_classified_positive_true_positive": total_annotated_classified_positive_true_positive,
        }
    else:
        n_classified_positive_annotated_by_area = np.random.multinomial(
            total_annotated_classified_positive,
            np.array(n_classified_positive_by_area)
            / sum(n_classified_positive_by_area),
        )
        assert (
            n_classified_positive_annotated_by_area.sum()
            == total_annotated_classified_positive
        )
        ps = np.array(n_images_by_area) - np.array(n_classified_positive_by_area)
        ps = ps / sum(ps)
        n_classified_negative_annotated_by_area = np.random.multinomial(
            total_annotated_classified_negative, ps
        )
        assert (
            n_classified_negative_annotated_by_area.sum()
            == total_annotated_classified_negative
        )
        n_classified_positive_annotated_positive_by_area = []
        n_classified_positive_annotated_negative_by_area = []
        n_classified_negative_annotated_positive_by_area = []
        n_classified_negative_annotated_negative_by_area = []
        n_non_annotated_by_area = []
        n_non_annotated_by_area_classified_positive = []
        for i in range(N):
            if n_classified_positive_annotated_by_area[i] > 0:
                vector_to_sample = [1] * n_true_positive_classified_positive_by_area[
                    i
                ] + [0] * n_true_negative_classified_positive_by_area[i]
                samples = np.random.choice(
                    vector_to_sample,
                    n_classified_positive_annotated_by_area[i],
                    replace=False,
                )
                assert len(samples) == n_classified_positive_annotated_by_area[i]
                n_classified_positive_annotated_positive_by_area.append(sum(samples))
                n_classified_positive_annotated_negative_by_area.append(
                    n_classified_positive_annotated_by_area[i] - sum(samples)
                )
            else:
                n_classified_positive_annotated_positive_by_area.append(0)
                n_classified_positive_annotated_negative_by_area.append(0)
            if n_classified_negative_annotated_by_area[i] > 0:
                vector_to_sample = [1] * n_true_positive_classified_negative_by_area[
                    i
                ] + [0] * n_true_negative_classified_negative_by_area[i]
                samples = np.random.choice(
                    vector_to_sample,
                    n_classified_negative_annotated_by_area[i],
                    replace=False,
                )
                assert len(samples) == n_classified_negative_annotated_by_area[i]
                n_classified_negative_annotated_positive_by_area.append(sum(samples))
                n_classified_negative_annotated_negative_by_area.append(
                    n_classified_negative_annotated_by_area[i] - sum(samples)
                )
            else:
                n_classified_negative_annotated_positive_by_area.append(0)
                n_classified_negative_annotated_negative_by_area.append(0)
            n_non_annotated_by_area.append(
                n_images_by_area[i]
                - n_classified_negative_annotated_by_area[i]
                - n_classified_positive_annotated_by_area[i]
            )
            n_non_annotated_by_area_classified_positive.append(
                n_classified_positive_by_area[i]
                - n_classified_positive_annotated_by_area[i]
            )

        observed_data = {
            "N": N,
            "N_edges": len(node1),
            "node1": node1,
            "node2": node2,
            "n_images_by_area": n_images_by_area,
            "n_classified_positive_by_area": n_classified_positive_by_area,
            "n_classified_positive_annotated_positive_by_area": n_classified_positive_annotated_positive_by_area,
            "n_classified_positive_annotated_negative_by_area": n_classified_positive_annotated_negative_by_area,
            "n_classified_negative_annotated_positive_by_area": n_classified_negative_annotated_positive_by_area,
            "n_classified_negative_annotated_negative_by_area": n_classified_negative_annotated_negative_by_area,
            "n_non_annotated_by_area": n_non_annotated_by_area,
            "n_non_annotated_by_area_classified_positive": n_non_annotated_by_area_classified_positive,
        }

    return {
        "observed_data": observed_data,
        "parameters": {
            "phi": phi,
            "phi_offset": phi_offset,
            "p_y_1_given_y_hat_1": p_y_1_given_y_hat_1,
            "p_y_1_given_y_hat_0": p_y_1_given_y_hat_0,
            "p_y_hat_1_given_y_1": p_y_hat_1_given_y_1,
            "p_y_hat_1_given_y_0": p_y_hat_1_given_y_0,
            "p_y": p_Y,
            "tau": tau,
            "alpha": alpha,
            "sigma": sigma,
        },
    }
