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


def read_real_data(fpath="flooding_ct_dataset.csv", annotations_have_locations=False, adj=[], adj_matrix_storage=False, use_external_covariates=False):
    single_compartment_for_debugging = False
    df = pd.read_csv(fpath)

    df[["total_images", "positive_images"]] = (
        df[["n_total", "n_classified_positive"]].astype(int).fillna(0)
    )
    N = len(df)
    n_images_by_area = df["total_images"].values
    n_classified_positive_by_area = df["positive_images"].values
    # store tract_id (the GEOID col of df) as an array of stan real numerics
    tract_id = df["GEOID"].astype(float).values


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

        # (9/20) CHECK UNDERLYING DATASET HERE

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
                #"W": adjm,
                # W_n is the total number of neighbor pairs, so sum of adjm 
                #"W_n": int(np.sum(adjm)),
                "N_edges": len(node1),
                "node1": node1,
                "node2": node2,
                "tract_id": tract_id,
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

            def process_covariates(df, observed_data, use_external_covariates=True):
                """Process covariates with robust handling for large N observations."""
                if not use_external_covariates:
                    return {"observed_data": observed_data}

                # Initialize binary flags
                df['any_311_report'] = False
                df['any_sensors'] = df['n_floodnet_sensors'] > 0

                # 1. Process right-skewed continuous variables
                skewed_cols = [
                    'ft_elevation_min', 
                    'ft_elevation_mean', 
                    'ft_elevation_max',
                    'dep_moderate_1_area', 
                    'dep_moderate_2_area'
                ]
                
                for col in skewed_cols:
                    # Ensure numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Check for outliers
                    data = df[col]
                    q1, q3 = np.percentile(data, [25, 75])
                    iqr = q3 - q1
                    outlier_mask = (data < q1 - 3*iqr) | (data > q3 + 3*iqr)
                    if np.any(outlier_mask):
                        print(f"Warning: {np.sum(outlier_mask)} outliers detected in {col}")
                    
                    df[f'{col}_log'] = np.log1p(df[col])
                
                # 2. Process categorical FSHRI (1-5)
                df['fshri'] = pd.to_numeric(df['fshri'], errors='coerce')
                fshri_dummies = pd.get_dummies(df['fshri'], prefix='fshri')
                
                for col in fshri_dummies.columns:
                    prop = fshri_dummies[col].mean()
                    if prop < 0.05:
                        print(f"Warning: Rare category detected in {col} with proportion {prop:.4f}")
                
                # 3. Collect covariates
                cols_to_use = [f'{col}_log' for col in skewed_cols]
                cols_to_use.append('any_sensors')
                
                # Process 311 binary columns
                for k in df.columns:
                    if '311' in k:
                        df[k] = pd.to_numeric(df[k], errors='coerce')
                        binary_col = (df[k] > 0).astype(int)
                        check_binary_frequency(binary_col)
                        df[f'binary_{k}'] = binary_col
                        print(f"Adding data from column {k}")
                        df['any_311_report'] = df['any_311_report'] | (df[k] > 0) 
                        cols_to_use.append(f'binary_{k}')
                
                # Convert to numeric matrix
                feature_matrix = df[cols_to_use].values.astype(float)
                fshri_matrix = fshri_dummies.values.astype(float)
                
                external_covariate_matrix = np.hstack([feature_matrix, fshri_matrix])
                
                # Z-score everything with robust checks
                n_features = external_covariate_matrix.shape[1]
                for i in range(n_features):
                    col_name = cols_to_use[i] if i < len(cols_to_use) else f"fshri_{i-len(cols_to_use)+1}"
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
                else:
                    observed_data['external_covariates'] = external_covariate_matrix
                
                observed_data['n_external_covariates'] = observed_data['external_covariates'].shape[1]
                print(f"Final external covariate matrix shape: {observed_data['external_covariates'].shape}")
                
                return {"observed_data": observed_data}
            
            return process_covariates(df, observed_data, use_external_covariates)
        else: 
            return {"observed_data": observed_data}
    else:
        return {
            "observed_data": {
                "N": N,
                "N_edges": len(node1),
                "node1": node1,
                "node2": node2,
                "tract_id": tract_id,
                "n_images_by_area": n_images_by_area,
                "n_classified_positive_by_area": n_classified_positive_by_area,
                "total_annotated_classified_negative": TOTAL_ANNOTATED_CLASSIFIED_NEGATIVE,
                "total_annotated_classified_positive": TOTAL_ANNOTATED_CLASSIFIED_POSITIVE,
                "total_annotated_classified_negative_true_positive": TOTAL_ANNOTATED_CLASSIFIED_NEGATIVE_TRUE_POSITIVE,
                "total_annotated_classified_positive_true_positive": TOTAL_ANNOTATED_CLASSIFIED_POSITIVE_TRUE_POSITIVE,
            }
        }


def validate_observed_data(observed_data, annotations_have_locations=False):
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

        # n_images_by_area - n_not_annotaed_by_area should equal the sum of n_classified_positive_annotated_positive_by_area, n_classified_positive_annotated_negative_by_area, n_classified_negative_annotated_positive_by_area, n_classified_negative_annotated_negative_by_area
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
                "n_images_by_area - n_not_annotaed_by_area should equal the sum of n_classified_positive_annotated_positive_by_area, n_classified_positive_annotated_negative_by_area, n_classified_negative_annotated_positive_by_area, n_classified_negative_annotated_negative_by_area"
            )

        # sum of n_classified_positive_annotated_positive_by_area and n_classified_positive_annotated_negative_by_area should equal TOTAL_ANNOTATED_CLASSIFIED_POSITIVE
        if (
            sum(observed_data["n_classified_positive_annotated_positive_by_area"])
            + sum(observed_data["n_classified_positive_annotated_negative_by_area"])
            != TOTAL_ANNOTATED_CLASSIFIED_POSITIVE
        ):
            raise ValueError(
                "sum of n_classified_positive_annotated_positive_by_area and n_classified_positive_annotated_negative_by_area should equal total_annotated_classified_positive"
            )

        # sum of n_classified_negative_annotated_positive_by_area and n_classified_negative_annotated_negative_by_area should equal TOTAL_ANNOTATED_CLASSIFIED_NEGATIVE
        if (
            sum(observed_data["n_classified_negative_annotated_positive_by_area"])
            + sum(observed_data["n_classified_negative_annotated_negative_by_area"])
            != TOTAL_ANNOTATED_CLASSIFIED_NEGATIVE
        ):
            raise ValueError(
                "sum of n_classified_negative_annotated_positive_by_area and n_classified_negative_annotated_negative_by_area should equal total_annotated_classified_negative"
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
