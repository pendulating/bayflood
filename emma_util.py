import pandas as pd
import stan
import numpy as np
from scipy.stats import pearsonr
from scipy.special import expit
import matplotlib.pyplot as plt
import arviz as az
import random

def read_real_data(single_compartment_for_debugging=False):
    df = pd.read_csv("flooding_ct_dataset.csv")
    df[['total_images','positive_images']] = df[['total_images','positive_images']].astype(int).fillna(0)
    N = len(df)
    n_images_by_area = df['total_images'].values
    n_classified_positive_by_area = df['positive_images'].values 

    # Generate adjacency matrix and neighborhood structure
    node1 = []
    node2 = []

    TOTAL_ANNOTATED_CLASSIFIED_NEGATIVE = 500
    TOTAL_ANNOTATED_CLASSIFIED_POSITIVE = 500
    TOTAL_ANNOTATED_CLASSIFIED_NEGATIVE_TRUE_POSITIVE = 3
    TOTAL_ANNOTATED_CLASSIFIED_POSITIVE_TRUE_POSITIVE = 329


    if single_compartment_for_debugging:
        N = 1
        n_images_by_area = [sum(n_images_by_area)]
        n_classified_positive_by_area = [sum(n_classified_positive_by_area)]

    return {'observed_data': {
                'N': N, 'N_edges': len(node1), 'node1': node1, 'node2': node2, 
                'n_images_by_area': n_images_by_area,
                'n_classified_positive_by_area': n_classified_positive_by_area, 
                'total_annotated_classified_negative': TOTAL_ANNOTATED_CLASSIFIED_NEGATIVE,
                'total_annotated_classified_positive': TOTAL_ANNOTATED_CLASSIFIED_POSITIVE,
                'total_annotated_classified_negative_true_positive': TOTAL_ANNOTATED_CLASSIFIED_NEGATIVE_TRUE_POSITIVE,
                'total_annotated_classified_positive_true_positive': TOTAL_ANNOTATED_CLASSIFIED_POSITIVE_TRUE_POSITIVE
            }
            }

def generate_simulated_data(N, images_per_location, 
                            total_annotated_classified_negative, 
                            total_annotated_classified_positive, 
                            icar_prior_setting):
    """
    Generate simulated data for the model.
    """    
    node1 = []
    node2 = []
    for i in range(N):
        for j in range(i+1, N):
            if np.random.rand() < 0.1:
                node1.append(i + 1) # one indexing for Stan. 
                node2.append(j + 1)
    phi_offset = random.random() * -3 - 1 # mean of phi.

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
    if icar_prior_setting != 'none':    
        phi = np.random.multivariate_normal(mean=np.zeros(N), cov=sigma)
    else:
        phi = np.random.normal(loc=0, size=N) # this uses no icar prior, just draws everything independently. 
    p_Y = expit(phi + phi_offset)
    n_images_by_area = np.random.poisson(images_per_location, N)
    p_y_hat_1_given_y_1 = random.random() * 0.5 + 0.2
    p_y_hat_1_given_y_0 = random.random() * 0.01 + 0.01

    n_classified_positive_by_area = []
    n_true_positive = []
    for i in range(N):
        n_true_positive.append(np.random.binomial(n_images_by_area[i], p_Y[i]))
        n_classified_positive_by_area.append(np.random.binomial(n_true_positive[-1], p_y_hat_1_given_y_1) + 
                                    np.random.binomial(n_images_by_area[i] - n_true_positive[-1], p_y_hat_1_given_y_0))
    empirical_p_yhat = sum(n_classified_positive_by_area) * 1.0 / sum(n_images_by_area)
    empirical_p_y = sum(n_true_positive) * 1.0 / sum(n_images_by_area)
    p_y_1_given_y_hat_1 = p_y_hat_1_given_y_1 * empirical_p_y / empirical_p_yhat
    p_y_1_given_y_hat_0 = (1 - p_y_hat_1_given_y_1) * empirical_p_y / (1 - empirical_p_yhat)
    print("empirical_p_y", empirical_p_y)
    print("empirical_p_yhat", empirical_p_yhat)
    print("p_y_hat_1_given_y_1", p_y_hat_1_given_y_1)
    print("p_y_hat_1_given_y_0", p_y_hat_1_given_y_0)
    print("p_y_1_given_y_hat_1", p_y_1_given_y_hat_1)
    print("p_y_1_given_y_hat_0", p_y_1_given_y_hat_0)
                     
    total_annotated_classified_negative_true_positive = np.random.binomial(total_annotated_classified_negative, p_y_1_given_y_hat_0)
    total_annotated_classified_positive_true_positive = np.random.binomial(total_annotated_classified_positive, p_y_1_given_y_hat_1)
    print("number of annotated classified negative which were positive: %i/%i" % (total_annotated_classified_negative_true_positive, total_annotated_classified_negative))
    print("number of annotated classified positive which were positive: %i/%i" % (total_annotated_classified_positive_true_positive, total_annotated_classified_positive))
    

    return {'observed_data':{'N':N, 'N_edges':len(node1), 'node1':node1, 'node2':node2, 
                             'n_images_by_area':n_images_by_area, 'n_classified_positive_by_area':n_classified_positive_by_area, 
                             'total_annotated_classified_negative':total_annotated_classified_negative,
                                'total_annotated_classified_positive':total_annotated_classified_positive,
                                'total_annotated_classified_negative_true_positive':total_annotated_classified_negative_true_positive,
                                'total_annotated_classified_positive_true_positive':total_annotated_classified_positive_true_positive},

            'parameters':{'phi':phi, 'phi_offset':phi_offset, 
                          'p_y_1_given_y_hat_1':p_y_1_given_y_hat_1,
                            'p_y_1_given_y_hat_0':p_y_1_given_y_hat_0, 
                            'p_y_hat_1_given_y_1':p_y_hat_1_given_y_1,
                            'p_y_hat_1_given_y_0':p_y_hat_1_given_y_0, 
                            'p_Y':p_Y, 
                            'tau':tau, 'alpha':alpha, 'sigma':sigma}}