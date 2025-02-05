# Dashcam-Bayesian Flooding Model Project 
# Developers: Matthew Franchi (mattwfranchi) and Emma Pierson (epierson9)
# Cornell Tech 

# In this script, we house a class that fits various Stan models to a processed dataset of urban street flooding conditions in New York City. 


## Module Imports 
import util
from IPython import embed


import json
from copy import deepcopy
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import datetime
import os
import logger
import multiprocessing
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("fork")

import pandas as pd
import stan
import numpy as np
from scipy.stats import pearsonr, spearmanr
import arviz as az
from scipy.special import expit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import nest_asyncio

import sys

import warnings
import argparse

from generate_maps import generate_maps
from refresh_cache import refresh_cache

from analysis_df import generate_nyc_analysis_df

LATEX_PLOTTING=False
if LATEX_PLOTTING:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

nest_asyncio.apply()


## Class Definition
class ICAR_MODEL:
    def __init__(
        self,
        PREFIX='',
        ICAR_PRIOR_SETTING="none",
        ANNOTATIONS_HAVE_LOCATIONS=True,
        EXTERNAL_COVARIATES=False,
        SIMULATED_DATA=False,
        ESTIMATE_PARAMS=[],
        EMPIRICAL_DATA_PATH="",
        adj=[],
        adj_matrix_storage=None,
        downsample_frac=1
    ):

        refresh_cache('mwf62')
        print(SIMULATED_DATA)

        # Sanity checks on user inputs 
        # EMPIRICAL_DATA_PATH should not be set if we are using simulated data
        if SIMULATED_DATA:
            assert EMPIRICAL_DATA_PATH == ""
        elif EMPIRICAL_DATA_PATH:
            assert not SIMULATED_DATA

        # adj_matrix_storage should be set if adj is set
        if adj:
            assert adj_matrix_storage is not None

        # if adj_matrix_storage is set, adj should be set
        # if adj_matrix_storage is False, adj should be a list of two string file paths
        # if adj_matrix_storage is True, adj should be a list of one string file path
        if adj_matrix_storage:
            assert adj
            assert isinstance(adj, list)
            if adj_matrix_storage is True:
                assert len(adj) == 1
                assert isinstance(adj[0], str)
            else:
                assert len(adj) == 2
                assert isinstance(adj[0], str)
                assert isinstance(adj[1], str)
            

        
        # This block of variables is fixed across modeling fitting runs, 
        # and represent metadata about real dataset, or simulated data
        
        # Real dataset metadata 
        self.N_ANNOTATED_CLASSIFIED_NEGATIVE = 500
        self.N_ANNOTATED_CLASSIFIED_POSITIVE = 500
        self.N_ANNOTATED_CLASSIFIED_NEGATIVE_TRUE_POSITIVE = 3
        self.N_ANNOTATED_CLASSIFIED_POSITIVE_TRUE_POSITIVE = 329
        self.TOTAL_PRED_POSITIVE = 1465
        self.TOTAL_PRED_NEGATIVE = 924747

        # Simulated data metadata
        self.N_SIMULATED_TRACTS = 1000

        # These flags control the behavior of the model fitting routine
        self.annotations_have_locations = ANNOTATIONS_HAVE_LOCATIONS
        self.use_simulated_data = SIMULATED_DATA
        self.use_external_covariates = EXTERNAL_COVARIATES
        self.downsample_frac = downsample_frac
        self.EMPIRICAL_DATA_PATH = EMPIRICAL_DATA_PATH

        self.icar_prior_setting = ICAR_PRIOR_SETTING
        assert self.icar_prior_setting in [
            "none",
            "icar",
            "proper",
            "just_model_p_y",
        ]

        self.VALID_ESTIMATE_PARAMETERS = ["p_y", "at_least_one_positive_image_by_area", "at_least_one_positive_image_by_area_if_you_have_100_images"]
        self.ADDITIONAL_PARAMS_TO_SAVE = []
        self.ESTIMATE_PARAMETERS = ESTIMATE_PARAMS
        for p in self.ESTIMATE_PARAMETERS:
            assert p in self.VALID_ESTIMATE_PARAMETERS


        # This dictionary stores the available stan models
        self.models = {
            "weighted_ICAR_prior": open("stan_models/weighted_ICAR_prior.stan").read(),
            "proper_car_prior": open("stan_models/proper_car_prior.stan").read(),
            "uniform_p_y": open(
                "stan_models/uniform_p_y_prior_just_for_debugging.stan"
            ).read(),
            "ICAR_prior_annotations_have_locations": open(
                "stan_models/ICAR_prior_annotations_have_locations.stan"
            ).read(),
        }

        self.logger = logger.setup_logger(f"ICAR_MODEL: {ICAR_PRIOR_SETTING}, ahl {ANNOTATIONS_HAVE_LOCATIONS}, simulated {SIMULATED_DATA}")
        self.logger.setLevel("INFO")
        self.logger.info("ICAR_MODEL instance initialized.")

        self.adj_path = adj
        self.adj_matrix_storage = adj_matrix_storage

        # other misc sanity checks 

        # cannot use the at_least_one_positive_image_by_area parameter if additional annotation location data is not utilized 
        if not self.annotations_have_locations: 
            assert 'at_least_one_positive_image_by_area' not in self.ESTIMATE_PARAMETERS


        # if there's a non-blank prefix, prepend it to runid 
        if PREFIX:
            self.logger.info(f"Setting prefix to {PREFIX}")
            self.RUNID = PREFIX
        else: 
            self.logger.info("No prefix set.")
            self.RUNID = ""

    def parse_data_for_validation(self):
        # write jsonified observed data to file for debugging
        # need to convert numpy arrays to lists
        observed_data_copy = self.data_to_use["observed_data"].copy()
        # observed_data_copy is a dict
        for k in observed_data_copy.keys():
            if isinstance(observed_data_copy[k], np.ndarray):
                observed_data_copy[k] = observed_data_copy[k].tolist()
            # serialized int64
            if isinstance(observed_data_copy[k], np.int64):
                observed_data_copy[k] = int(observed_data_copy[k])
            # serialize nd arrays with int 64 elements
            if isinstance(observed_data_copy[k], list):
                for i in range(len(observed_data_copy[k])):
                    if isinstance(observed_data_copy[k][i], np.int64):
                        observed_data_copy[k][i] = int(observed_data_copy[k][i])

        self.logger.info(
            "Successfully converted the observed data into numpy arrays for inspection."
        )

        return observed_data_copy

    def load_data(self):
        if self.use_simulated_data:
            self.logger.info("Generating simulated data.")
            N = self.N_SIMULATED_TRACTS
            self.data_to_use = util.generate_simulated_data(
                N=N,
                images_per_location=1000,
                total_annotated_classified_negative=self.N_ANNOTATED_CLASSIFIED_NEGATIVE,
                total_annotated_classified_positive=self.N_ANNOTATED_CLASSIFIED_POSITIVE,
                icar_prior_setting=self.icar_prior_setting,
                annotations_have_locations=self.annotations_have_locations,
            )

            if self.downsample_frac < 1:
                self.logger.info("Downsampling data.")
                self.data_to_use = self.downsample_data(self.data_to_use, downsample_frac=self.downsample_frac)

            self.logger.success("Successfully generated simulated data.")
        else:
            self.logger.info("Reading empirical data.")
            self.data_to_use, external_covariates_info = util.read_real_data(fpath=self.EMPIRICAL_DATA_PATH,
                annotations_have_locations=self.annotations_have_locations, adj=self.adj_path, adj_matrix_storage=self.adj_matrix_storage, 
                                use_external_covariates = self.use_external_covariates
            )

            if self.use_external_covariates:
                # write external covariates to file for debugging
                print(external_covariates_info)
                external_covariates_info = pd.DataFrame.from_dict(external_covariates_info['external_covariates'])
                with open(f"runs/{self.RUNID}/external_covariates.csv", "w") as f:
                    external_covariates_info.to_csv(f)
                    
            self.logger.success("Successfully read empirical data.")

            if self.downsample_frac < 1:
                self.logger.info("Downsampling data.")
                self.data_to_use = self.downsample_data(self.data_to_use, downsample_frac=self.downsample_frac)

            # validate observed data
            observed_data_copy = self.parse_data_for_validation()
            util.validate_observed_data(
                observed_data_copy, self.annotations_have_locations, self.downsample_frac
            )
            self.logger.success("Successfully validated the observed data.")
            del observed_data_copy



    def fit(self, CYCLES=1, WARMUP=1000, SAMPLES=1500, data_already_loaded=False):
        # pass in data_already_loaded = True if you want to use data that's already been loaded in. 
        # by default the method reloads the data. 
        self.RUNID = self.RUNID + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M")

        # add parent dirs that split runs based on simulated or empirical, annotations_have_locations, and icar_prior_setting
        self.RUNID = f"icar_{self.icar_prior_setting}/simulated_{self.use_simulated_data}/ahl_{self.annotations_have_locations}/covariates_{self.use_external_covariates}/{self.RUNID}"

        os.makedirs(f"runs/{self.RUNID}", exist_ok=True)

        for i in range(CYCLES):
            if not data_already_loaded:
                self.load_data()

            if self.icar_prior_setting == "icar":
                self.logger.info("Building model with ICAR prior.")
                self.data_to_use["observed_data"]["use_ICAR_prior"] = 1
                if self.annotations_have_locations:
                    self.logger.info(
                        "Building model with annotations have locations."
                    )
                    self.logger.info("Building model with use_external_covariates = %s" % self.use_external_covariates)
                    model = stan.build(
                        self.models["ICAR_prior_annotations_have_locations"],
                        data=self.data_to_use["observed_data"],
                    )
                    self.ADDITIONAL_PARAMS_TO_SAVE += ['spatial_sigma', 'external_covariate_beta']

                else:
                    self.logger.info("Building model without annotation location data.")
                    model = stan.build(
                        self.models["weighted_ICAR_prior"],
                        data=self.data_to_use["observed_data"],
                    )
            elif self.icar_prior_setting == "none":
                self.logger.info("Building model with no ICAR prior.")
                self.data_to_use["observed_data"]["use_ICAR_prior"] = 0
                if self.annotations_have_locations:
                    self.logger.info(
                        "Building model with annotations have locations."
                    )
                    self.logger.info("Building model with use_external_covariates = %s" % self.use_external_covariates)
                    model = stan.build(
                        self.models["ICAR_prior_annotations_have_locations"],
                        data=self.data_to_use["observed_data"],
                    )
                    self.ADDITIONAL_PARAMS_TO_SAVE +=  ['spatial_sigma', 'external_covariate_beta']

                    
                else:
                    self.logger.info("Building model without annotation location data.")
                    model = stan.build(
                        self.models["weighted_ICAR_prior"],
                        data=self.data_to_use["observed_data"],
                    )
            elif self.icar_prior_setting == "just_model_p_y":
                del self.data_to_use["observed_data"]["node1"]
                del self.data_to_use["observed_data"]["node2"]
                del self.data_to_use["observed_data"]["N_edges"]
                self.logger.info("Building model with just model p_y.")
                model = stan.build(
                    self.models["uniform_p_y"], data=self.data_to_use["observed_data"]
                )

            elif self.icar_prior_setting == "proper":
                model = stan.build(self.models['proper_car_prior'], data=self.data_to_use['observed_data'])
            

            else:
                raise ValueError("Invalid icar_prior_options", self.icar_prior_setting)


            self.logger.info(f"Successfully built the model, with use_icar_prior: {self.data_to_use['observed_data']['use_ICAR_prior']}.")

            with warnings.catch_warnings(action="ignore"):
                fit = model.sample(num_chains=4, num_warmup=WARMUP, num_samples=SAMPLES)
            print(az.summary(fit))
            df = fit.to_frame()

            self.logger.success("Successfully sampled the model.")

            # write metadata to file 
            # ANNOTATIONS_HAVE_LOCATIONS, SIMULATED_DATA, CYCLES, WARMUP, SAMPLES, use_icar_prior, icar_prior_weight, icar_prior_setting
            # N_ANNOTATED_CLASSIFIED_NEGATIVE, N_ANNOTATED_CLASSIFIED_POSITIVE, N_ANNOTATED_CLASSIFIED_NEGATIVE_TRUE_POSITIVE, N_ANNOTATED_CLASSIFIED_POSITIVE_TRUE_POSITIVE, TOTAL_PRED_POSITIVE, TOTAL_PRED_NEGATIVE, N_SIMULATED_TRACTS
            # self.adj, self.adj_matrix_storage
            metadata = {
                "RUNID": self.RUNID,
                "ANNOTATIONS_HAVE_LOCATIONS": self.annotations_have_locations,
                "SIMULATED_DATA": self.use_simulated_data,
                "EXTERNAL_COVARIATES": self.use_external_covariates,
                "CYCLES": CYCLES,
                "WARMUP": WARMUP,
                "SAMPLES": SAMPLES,
                "use_icar_prior": self.data_to_use["observed_data"]["use_ICAR_prior"],
                "icar_prior_setting": self.icar_prior_setting,
                "N_ANNOTATED_CLASSIFIED_NEGATIVE": self.N_ANNOTATED_CLASSIFIED_NEGATIVE,
                "N_ANNOTATED_CLASSIFIED_POSITIVE": self.N_ANNOTATED_CLASSIFIED_POSITIVE,
                "N_ANNOTATED_CLASSIFIED_NEGATIVE_TRUE_POSITIVE": self.N_ANNOTATED_CLASSIFIED_NEGATIVE_TRUE_POSITIVE,
                "N_ANNOTATED_CLASSIFIED_POSITIVE_TRUE_POSITIVE": self.N_ANNOTATED_CLASSIFIED_POSITIVE_TRUE_POSITIVE,
                "TOTAL_PRED_POSITIVE": self.TOTAL_PRED_POSITIVE,
                "TOTAL_PRED_NEGATIVE": self.TOTAL_PRED_NEGATIVE,
                "N_SIMULATED_TRACTS": self.N_SIMULATED_TRACTS,
                "adj": self.adj_path,
                "adj_matrix_storage": self.adj_matrix_storage,

            }

            with open(f"runs/{self.RUNID}/metadata.json", "w") as f:
                # write with a new line between each key-value pair
                f.write(json.dumps(metadata, indent=4))


            return fit, df
        
    def divide_data_into_train_and_test_set(self, full_dataset, train_frac=0.7):
        """
        Partitions the images into a train and test set. For each Census tract, randomly 
        assigns a fraction of the images to the train set, and the rest to the test set.
        This is a bit tricky to do because the raw data comes as counts. 
        """
        train_data = {}
        test_data = {}
        full_dataset = deepcopy(full_dataset)
        # add a convenience field because it makes the rest of the code easier to write succinctly. 
        full_dataset['observed_data']['n_non_annotated_by_area_classified_negative'] = full_dataset['observed_data']['n_non_annotated_by_area'] - full_dataset['observed_data']['n_non_annotated_by_area_classified_positive']
        for k in full_dataset['observed_data']:
            if k in ['N', 'N_edges', 'node1', 'node2', 'tract_id', 'center_of_phi_offset_prior', 'external_covariates', 'n_external_covariates']:
                train_data[k] = deepcopy(full_dataset['observed_data'][k])
                test_data[k] = deepcopy(full_dataset['observed_data'][k])
        for k in ['n_classified_positive_annotated_positive_by_area', 
                  'n_classified_positive_annotated_negative_by_area', 
                  'n_classified_negative_annotated_negative_by_area', 
                  'n_classified_negative_annotated_positive_by_area', 
                  'n_non_annotated_by_area_classified_positive', 
                  'n_non_annotated_by_area_classified_negative']:
                train_data[k] = np.random.binomial(full_dataset['observed_data'][k], train_frac)
                test_data[k] = full_dataset['observed_data'][k] - train_data[k]
                assert (train_data[k] >= 0).all()
                assert (test_data[k] >= 0).all()
        train_data['n_non_annotated_by_area'] = train_data['n_non_annotated_by_area_classified_positive'] + train_data['n_non_annotated_by_area_classified_negative']
        test_data['n_non_annotated_by_area'] = test_data['n_non_annotated_by_area_classified_positive'] + test_data['n_non_annotated_by_area_classified_negative']
        train_data['n_images_by_area'] = train_data['n_non_annotated_by_area'] + train_data['n_classified_positive_annotated_positive_by_area'] + train_data['n_classified_positive_annotated_negative_by_area'] + train_data['n_classified_negative_annotated_negative_by_area'] + train_data['n_classified_negative_annotated_positive_by_area']
        test_data['n_images_by_area'] = test_data['n_non_annotated_by_area'] + test_data['n_classified_positive_annotated_positive_by_area'] + test_data['n_classified_positive_annotated_negative_by_area'] + test_data['n_classified_negative_annotated_negative_by_area'] + test_data['n_classified_negative_annotated_positive_by_area']
        train_data['n_classified_positive_by_area'] = train_data['n_classified_positive_annotated_positive_by_area'] + train_data['n_classified_positive_annotated_negative_by_area'] + train_data['n_non_annotated_by_area_classified_positive']
        test_data['n_classified_positive_by_area'] = test_data['n_classified_positive_annotated_positive_by_area'] + test_data['n_classified_positive_annotated_negative_by_area'] + test_data['n_non_annotated_by_area_classified_positive']

        for k in full_dataset['observed_data'].keys():
            if k not in ['N', 'N_edges', 'node1', 'node2', 'tract_id', 'center_of_phi_offset_prior', 'external_covariates', 'n_external_covariates']:
                assert (train_data[k] + test_data[k] == full_dataset['observed_data'][k]).all()
        print("With a train frac of %2.3f, train set has %i total images; test set has %i" % 
                (train_frac, train_data['n_images_by_area'].sum(), test_data['n_images_by_area'].sum()))
        return train_data, test_data

    def downsample_data(self, full_dataset, downsample_frac=0.1):
        """
        Downsamples only the annotated images in the dataset by downsample_frac.
        Non-annotated images are left unchanged.
        
        Parameters:
        -----------
        full_dataset : dict
            Input dataset containing observed data
        downsample_frac : float
            Fraction of annotated images to keep (default: 0.1)
            
        Returns:
        --------
        dict : Modified dataset with downsampled annotated images
        """
        downsampled_data = deepcopy(full_dataset)

        # add a convenience field with the total number of annotated images
        full_dataset['observed_data']['n_annotated_by_area'] = (
            full_dataset['observed_data']['n_classified_positive_annotated_positive_by_area'] +
            full_dataset['observed_data']['n_classified_positive_annotated_negative_by_area'] +
            full_dataset['observed_data']['n_classified_negative_annotated_negative_by_area'] +
            full_dataset['observed_data']['n_classified_negative_annotated_positive_by_area']
        )
        
        # Annotated image fields that we'll downsample
        annotated_fields = [
            'n_classified_positive_annotated_positive_by_area',
            'n_classified_positive_annotated_negative_by_area',
            'n_classified_negative_annotated_negative_by_area',
            'n_classified_negative_annotated_positive_by_area'
        ]
        
        # Downsample annotated images
        for k in annotated_fields:
            downsampled_data['observed_data'][k] = np.random.binomial(
                full_dataset['observed_data'][k], 
                downsample_frac
            )
            assert (downsampled_data['observed_data'][k] >= 0).all()
        
        # Update derived fields
        downsampled_data['observed_data']['n_annotated_by_area'] = (
            downsampled_data['observed_data']['n_classified_positive_annotated_positive_by_area'] + 
            downsampled_data['observed_data']['n_classified_positive_annotated_negative_by_area'] + 
            downsampled_data['observed_data']['n_classified_negative_annotated_negative_by_area'] + 
            downsampled_data['observed_data']['n_classified_negative_annotated_positive_by_area']
        )
        
        # Total images count (non-annotated + downsampled annotated)
        downsampled_data['observed_data']['n_images_by_area'] = (
            downsampled_data['observed_data']['n_non_annotated_by_area'] +
            downsampled_data['observed_data']['n_annotated_by_area']
        )
        
        # Update positive classifications count
        downsampled_data['observed_data']['n_classified_positive_by_area'] = (
            downsampled_data['observed_data']['n_classified_positive_annotated_positive_by_area'] +
            downsampled_data['observed_data']['n_classified_positive_annotated_negative_by_area'] +
            downsampled_data['observed_data']['n_non_annotated_by_area_classified_positive']
        )

        # update total_annotated_classified_positive and total_annotated_classified_negative
        downsampled_data['observed_data']['total_annotated_classified_positive'] = (
            downsampled_data['observed_data']['n_classified_positive_annotated_positive_by_area'] +
            downsampled_data['observed_data']['n_classified_positive_annotated_negative_by_area']
        )

        downsampled_data['observed_data']['total_annotated_classified_negative'] = (
            downsampled_data['observed_data']['n_classified_negative_annotated_positive_by_area'] +
            downsampled_data['observed_data']['n_classified_negative_annotated_negative_by_area']
        )


        
        
        # Print summary statistics
        self.logger.info(f"Original annotated images: {full_dataset['observed_data']['n_annotated_by_area'].sum()}")
        self.logger.info(f"Downsampled annotated images: {downsampled_data['observed_data']['n_annotated_by_area'].sum()}")
        self.logger.info(f"Total images after downsampling: {downsampled_data['observed_data']['n_images_by_area'].sum()}")
        
        return downsampled_data
        
    def construct_graph_laplacian_baseline(self, N, N_edges, node1, node2, y, alpha=0.01, iterations=1):
        # https://www.math.fsu.edu/~bertram/lectures/Diffusion.pdf and ChatGPT seem to agree on this. 
        y = deepcopy(y)
        A = np.zeros((N, N))
        A[node1 - 1, node2 - 1] = 1
        A[node2 - 1, node1 - 1] = 1
        assert A.sum() == 2 * N_edges == 2 * len(node1) == 2 * len(node2)
        assert (node1 != node2).all()
        assert (A == (A.T)).all()
        degrees = A.sum(axis=1)
        D = np.diag(degrees)
        L = D - A
        for _ in range(iterations):
            assert (L@y).shape == y.shape
            y = y - alpha * (L @ y)
        return y


    def extract_baselines(self, data):
        """
        extracts various simple baselines from the data. 
        We actually end up running this on both the train set (where it's genuinely used to create baselines)
        and the test set (where it's used to create ground-truth measures to validate against). 
        """
        
        frac_positive_classifications_baseline = data['n_classified_positive_by_area'] / data['n_images_by_area']
        is_na = np.isnan(frac_positive_classifications_baseline)
        print("warning: fraction %2.3f entries of frac_positive_classifications_baseline are NA; imputing with mean" % (is_na.mean()))
        frac_positive_classifications_baseline[is_na] = 1. * data['n_classified_positive_by_area'].sum() / data['n_images_by_area'].sum()
        # not including fraction of positives among ground truth for now because 
        # there are too many NAs and it's not clear to me what the appropriate thing to fill that in with is. 


        # also include some more sophisticated ML methods that use the graph laplacian. 
        graph_laplacian_frac_pos_classifications_one_iter = self.construct_graph_laplacian_baseline(N=data['N'], N_edges=data['N_edges'], node1=np.array(data['node1']), node2=np.array(data['node2']), 
                                                                                    y=frac_positive_classifications_baseline, iterations=1)
        graph_laplacian_frac_pos_classifications_five_iter = self.construct_graph_laplacian_baseline(N=data['N'], N_edges=data['N_edges'], node1=np.array(data['node1']), node2=np.array(data['node2']),
                                                                                    y=frac_positive_classifications_baseline, iterations=5)
        
        graph_laplacian_n_positive_ground_truth_one_iter = self.construct_graph_laplacian_baseline(N=data['N'], N_edges=data['N_edges'], node1=np.array(data['node1']), node2=np.array(data['node2']), 
                                                                                    y=data['n_classified_positive_annotated_positive_by_area'] + data['n_classified_negative_annotated_positive_by_area'], iterations=1)
        graph_laplacian_n_positive_ground_truth_five_iter = self.construct_graph_laplacian_baseline(N=data['N'], N_edges=data['N_edges'], node1=np.array(data['node1']), node2=np.array(data['node2']),
                                                                                    y=data['n_classified_positive_annotated_positive_by_area'] + data['n_classified_negative_annotated_positive_by_area'], iterations=5)
        # supervised baselines which predict outcome from external covariates
        assert (data['external_covariates'][:, 0] == 1).all()
        X = data['external_covariates'][:, 1:]
        OLS_pred_frac_positive_classifications = LinearRegression().fit(X, frac_positive_classifications_baseline).predict(X)
        OLS_pred_n_positive_ground_truth = LinearRegression().fit(X, data['n_classified_positive_annotated_positive_by_area'] + data['n_classified_negative_annotated_positive_by_area']).predict(X)
        RandomForest_pred_frac_positive_classifications = RandomForestRegressor(random_state=777).fit(X, frac_positive_classifications_baseline).predict(X)
        RandomForest_pred_n_positive_ground_truth = RandomForestRegressor(random_state=777).fit(X, data['n_classified_positive_annotated_positive_by_area'] + data['n_classified_negative_annotated_positive_by_area']).predict(X)
        
        estimates = {# heuristic baselines
                    'frac_positive_classifications':frac_positive_classifications_baseline, 
                     'any_positive_classifications': 1. * (data['n_classified_positive_by_area'] > 0), 
                     'n_positive_classifications':data['n_classified_positive_by_area'],
                     'any_positive_ground_truth':1. * ((data['n_classified_positive_annotated_positive_by_area'] + data['n_classified_negative_annotated_positive_by_area']) > 0), 
                     'n_positive_ground_truth':data['n_classified_positive_annotated_positive_by_area'] + data['n_classified_negative_annotated_positive_by_area'],
                    # graph laplacian baselines
                     'graph_laplacian_frac_pos_classifications_one_iter':graph_laplacian_frac_pos_classifications_one_iter,
                        'graph_laplacian_frac_pos_classifications_five_iter':graph_laplacian_frac_pos_classifications_five_iter,
                        'graph_laplacian_n_positive_ground_truth_one_iter':graph_laplacian_n_positive_ground_truth_one_iter, 
                        'graph_laplacian_n_positive_ground_truth_five_iter':graph_laplacian_n_positive_ground_truth_five_iter,
                        # supervised learning baselines
                        'OLS_pred_frac_positive_classifications':OLS_pred_frac_positive_classifications,
                        'OLS_pred_n_positive_ground_truth':OLS_pred_n_positive_ground_truth, 
                        'RandomForest_pred_frac_positive_classifications':RandomForest_pred_frac_positive_classifications,
                        'RandomForest_pred_n_positive_ground_truth':RandomForest_pred_n_positive_ground_truth
                     }
        
        return estimates
        
    def compare_to_baselines(self, train_frac=0.2, save=True):
        """
        fit on train set, assess on test set, compare to baselines estimated in extract_baselines. 
        """
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_rows', 50)
        pd.set_option('display.max_columns', 50)
        self.load_data()
        train_data, test_data = self.divide_data_into_train_and_test_set(self.data_to_use, train_frac=train_frac)
        method_and_baselines = self.extract_baselines(train_data)
        ground_truth = self.extract_baselines(test_data)

        self.data_to_use = {'observed_data':train_data}
        fit, df = self.fit(CYCLES=1, WARMUP=8000, SAMPLES=8000, data_already_loaded=True)

        self.plot_results(fit, df)

        p_y_bayesian_estimate = np.array([df['p_y.%i' % i].mean() for i in range(1, train_data['N'] + 1)])
        at_least_one_positive_by_area_bayesian_estimate = np.array([df['at_least_one_positive_image_by_area.%i' % i].mean() for i in range(1, train_data['N'] + 1)])
        
        method_and_baselines['bayesian_model_p_y'] = p_y_bayesian_estimate
        method_and_baselines['bayesian_model_at_least_one_positive_by_area'] = at_least_one_positive_by_area_bayesian_estimate
        
        performance = {}
        no_images_in_test = test_data['n_images_by_area'] == 0
        print("warning: test set has fraction %2.3f tracts with no images; not using these in evals" % no_images_in_test.mean())
        for estimate in method_and_baselines:
            performance[estimate] = {}
            performance[estimate]['pearson r, frac_positive_classifications'] = pearsonr(method_and_baselines[estimate][~no_images_in_test], ground_truth['frac_positive_classifications'][~no_images_in_test])[0]
            performance[estimate]['AUC, any ground truth positive'] = roc_auc_score(ground_truth['any_positive_ground_truth'][~no_images_in_test], method_and_baselines[estimate][~no_images_in_test])
            performance[estimate]['AUC, any classified positive'] = roc_auc_score(ground_truth['any_positive_classifications'][~no_images_in_test], method_and_baselines[estimate][~no_images_in_test])
        print(pd.DataFrame(performance).transpose())

        if save: 
            self.logger.info(f"Saving performance csv to runs/{self.RUNID}/performance_on_baselines.csv")
            pd.DataFrame(performance).transpose().to_csv(
                f"runs/{self.RUNID}/performance_on_baselines.csv"
            )

            
        return performance

    def plot_results(self, fit, df):

        def validate_results(summary, rhat_thres=1.1):
            """
            Validate the results of the fit by checking that the rhat values are below a certain threshold.
            summary is a pandas DataFrame with a column 'r_hat'
            """
            
            # warning log any rhat values above the threshold
            for i, row in summary.iterrows():
                if row['r_hat'] > rhat_thres:
                    self.logger.warning(f"r_hat for parameter {i} is {row['r_hat']}, above threshold of {rhat_thres}")
    

        def print_write_results(fit):
            summary = az.summary(
                fit,
                var_names=[
                    "p_y_hat_1_given_y_1",
                    "p_y_hat_1_given_y_0",
                    #"p_y_1_given_y_hat_1",
                    #"p_y_1_given_y_hat_0",
                    #"empirical_p_yhat",
                ] + self.ESTIMATE_PARAMETERS + self.ADDITIONAL_PARAMS_TO_SAVE,
            )

            print(summary)

            # validate the summary 
            validate_results(summary)

            # also, write to file 
            with open(f"runs/{self.RUNID}/summary.txt", "w") as f:
                f.write(
                    summary.to_string()
                )
        
        print_write_results(fit)

        if self.use_simulated_data:

            for p in self.ESTIMATE_PARAMETERS:

                if p == "p_y":
                    # new figure 
                    plt.figure(figsize=[6,6])
                    estimate = [
                        df[f"p_y.{i}"].mean() for i in range(1, self.N_SIMULATED_TRACTS + 1)
                    ]
                    plt.scatter(self.data_to_use["parameters"]["p_y"], estimate)
                    plt.title(
                        "True vs. inferred p, r = %.2f"
                        % pearsonr(self.data_to_use["parameters"][p], estimate)[0]
                    )
                    max_val = max(max(self.data_to_use["parameters"][p]), max(estimate))
                    plt.xlabel("True p")
                    plt.ylabel("Inferred p")
                    plt.plot([0, max_val], [0, max_val], "r--")
                    plt.xlim([0, max_val])
                    plt.ylim([0, max_val])

                    plt.savefig(f"runs/{self.RUNID}/true_vs_inferred_p.png")
                    plt.close()

                    # plot histogram
                    if self.icar_prior_setting == "proper":
                        param_names = [
                            "p_y_hat_1_given_y_1",
                            "p_y_hat_1_given_y_0",
                            "p_y_1_given_y_hat_1",
                            "p_y_1_given_y_hat_0",
                            "phi_offset",
                            "alpha",
                            "tau",
                        ]
                    elif self.icar_prior_setting == "just_model_p_y":
                        param_names = [
                            "p_y_hat_1_given_y_1",
                            "p_y_hat_1_given_y_0",
                            "p_y_1_given_y_hat_1",
                            "p_y_1_given_y_hat_0",
                            "phi_offset",
                        ]
                    else:
                        param_names = [
                            "p_y_hat_1_given_y_1",
                            "p_y_hat_1_given_y_0",
                            "p_y_1_given_y_hat_1",
                            "p_y_1_given_y_hat_0",
                        ]

                    # new figure 
                    plt.figure(figsize=[12, 3])
                    for k in param_names:
                        plt.subplot(1, len(param_names), param_names.index(k) + 1)
                        # histogram of posterior samples
                        plt.hist(df[k], bins=50, density=True)
                        plt.title(k)
                        plt.axvline(self.data_to_use["parameters"][k], color="red")
                    plt.savefig(f"runs/{self.RUNID}/simulated_params_histogram_{p}.png")
                    plt.close()

        else:

            for p in self.ESTIMATE_PARAMETERS:

                if p == "p_y":

                    empirical_estimate = (
                        self.data_to_use["observed_data"]["n_classified_positive_by_area"]
                        / self.data_to_use["observed_data"]["n_images_by_area"]
                    )
                    print(
                        "Warning: %i of %i empirical p_yhat values are 0; these are being ignored"
                        % (sum(np.isnan(empirical_estimate)), len(empirical_estimate))
                    )

                    self.logger.info(
                        f"Using {', '.join(self.ESTIMATE_PARAMETERS)} as estimate parameters."
                    )
                    estimate = np.array(
                        [
                            df[f"p_y.{i}"].mean()
                            for i in range(1, len(empirical_estimate) + 1)
                        ]
                    )
                    estimate_CIs = [
                        df[f"p_y.{i}"].quantile([0.025, 0.975])
                        for i in range(1, len(empirical_estimate) + 1)
                    ]
                    n_images_by_area = self.data_to_use["observed_data"]["n_images_by_area"]
                    # make errorbar plot
                    image_cutoff = 100

                    plt.errorbar(
                        empirical_estimate[n_images_by_area >= image_cutoff],
                        estimate[n_images_by_area >= image_cutoff],
                        yerr=np.array(estimate_CIs)[n_images_by_area >= image_cutoff].T,
                        fmt="o",
                        color="blue",
                        ecolor="lightgray",
                        elinewidth=1,
                        capsize=3,
                        alpha=0.5,
                        label="n_images_by_area >= %i" % image_cutoff,
                    )

                    plt.errorbar(
                        empirical_estimate[n_images_by_area < image_cutoff],
                        estimate[n_images_by_area < image_cutoff],
                        yerr=np.array(estimate_CIs)[n_images_by_area < image_cutoff].T,
                        fmt="o",
                        color="red",
                        ecolor="lightgray",
                        elinewidth=1,
                        capsize=3,
                        alpha=0.5,
                        label="n_images_by_area < %i" % image_cutoff,
                    )

                    plt.legend()

                    # plot prior on p_y as vertical line.
                    #prior_on_p_y = expit(df["phi_offset"]).mean()
                    #plt.axhline(expit(prior_on_p_y), color="black", linestyle="--")
                    is_nan = np.isnan(empirical_estimate)
                    plt.title(
                        rf"Corr. between empirical $p_{{\hat{{y}}}}$ and $p_y$, r={pearsonr(empirical_estimate[~is_nan], estimate[~is_nan])[0]:.2f}")
                    
                    plt.xlabel(r"empirical $p(y = 1)$")
                    plt.ylabel(r"inferred $p(y = 1)$")
                    # logarithmic axes
                    plt.xscale("log")
                    plt.yscale("log")
                    plt.savefig(f"runs/{self.RUNID}/empirical_vs_inferred_p.png")
                    plt.close()

    def plot_histogram(self, fit, df):

        for p in self.ESTIMATE_PARAMETERS:
            assert p in self.VALID_ESTIMATE_PARAMETERS

            # histogram of parameter
            empirical_estimate = (
                self.data_to_use["observed_data"]['n_classified_positive_by_area']
                / self.data_to_use["observed_data"]["n_images_by_area"]
            )
            fig, ax = plt.subplots()
            estimate = np.array(
                [
                    df[f"{p}.{i}"].mean()
                    for i in range(1, len(empirical_estimate) + 1)
                ]
            )
            ax.hist(estimate, bins=200)
            ax.set_title(f"Probability distribution - {p}")
            ax.set_xlabel(f"{p}")
            ax.set_ylabel("Density")
            plt.savefig(f"runs/{self.RUNID}/histogram_{p}.png")
            plt.close()

    def plot_scatter(self, fit, df):
        for p in self.ESTIMATE_PARAMETERS:
            assert p in self.VALID_ESTIMATE_PARAMETERS

            # scatter plot of parameter
            empirical_estimate = (
                self.data_to_use["observed_data"]['n_classified_positive_by_area']
                / self.data_to_use["observed_data"]["n_images_by_area"]
            )
            estimate = np.array(
                [
                    df[f"{p}.{i}"].mean()
                    for i in range(1, len(empirical_estimate) + 1)
                ]
            )
            fig, ax = plt.subplots()
            ax.scatter(empirical_estimate, estimate)
            ax.set_title(f"Scatter plot - {p}")
            ax.set_xlabel(f"Empirical {p}")
            ax.set_ylabel(f"Inferred {p}")
            plt.savefig(f"runs/{self.RUNID}/scatter_{p}.png")
            plt.close()


    def write_estimate(self, fit, df):


        for p in self.ESTIMATE_PARAMETERS:
            assert p in self.VALID_ESTIMATE_PARAMETERS

            empirical_estimate = (
                self.data_to_use["observed_data"]["n_classified_positive_by_area"]
                / self.data_to_use["observed_data"]["n_images_by_area"]
            )

            estimate = np.array(
                [
                    df[f"{p}.{i}"].mean()
                    for i in range(1, len(empirical_estimate) + 1)
                ]
            )

            estimate_CIs = [
                df[f"{p}.{i}"].quantile([0.025, 0.975])
                for i in range(1, len(empirical_estimate) + 1)
            ]

            n_images_by_area = self.data_to_use["observed_data"]["n_images_by_area"]
            tract_id = self.data_to_use["observed_data"]["tract_id"]

            # make df to write
            results = pd.DataFrame(
                {
                    "tract_id": tract_id,
                    "empirical_estimate": empirical_estimate,
                    p: estimate,
                    f"{p}_CI_lower": np.array(estimate_CIs)[:, 0],
                    f"{p}_CI_upper": np.array(estimate_CIs)[:, 1],
                    "n_images_by_area": n_images_by_area,
                }
            )

            results.to_csv(
                f"runs/{self.RUNID}/estimate_{p}.csv", index=False
            )


if __name__ == "__main__":

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process some settings.")

    # Add required argument for ICAR prior setting
    parser.add_argument(
        "icar_prior_setting", 
        type=str, 
        help="The setting for the ICAR prior."
    )

    # Boolean flags
    parser.add_argument(
        "--annotations_have_locations",
        action='store_true',
        default=False,
        help="Include if annotations have associated locations"
    )

    parser.add_argument(
        "--simulated_data",
        action='store_true',
        default=False,
        help="Include if using simulated data"
    )

    parser.add_argument(
        '--external_covariates',
        action='store_true',
        default=False,
        help="Include if using external covariates"
    )

    parser.add_argument(
        '--compare_to_baselines',
        action='store_true',
        default=False,
        help='Include to run comparisons against baselines'
    )

    # Prefix argument
    parser.add_argument(
        '--prefix',
        type=str,
        required=False,
        help='Prefix for the run ID when saving results'
    )

    # downsampling argument 
    parser.add_argument(
        '--downsample_frac', 
        action='store',
        type=float,
        default=1,
        help='Fraction of annotated images to keep in the dataset'
    )



    # Parse the arguments
    args = parser.parse_args()

    model = ICAR_MODEL(
            PREFIX=args.prefix,
            ICAR_PRIOR_SETTING=args.icar_prior_setting,
            ESTIMATE_PARAMS=["p_y", "at_least_one_positive_image_by_area", "at_least_one_positive_image_by_area_if_you_have_100_images"],
            ANNOTATIONS_HAVE_LOCATIONS=args.annotations_have_locations,
            EXTERNAL_COVARIATES=args.external_covariates,
            SIMULATED_DATA=args.simulated_data,
            EMPIRICAL_DATA_PATH="aggregation/context_df_02052025.csv",
            adj=["data/processed/ct_nyc_adj_list_custom_geometric_node1.txt","data/processed/ct_nyc_adj_list_custom_geometric_node2.txt"],
            adj_matrix_storage=False,
            downsample_frac=args.downsample_frac
        )

    if args.compare_to_baselines:
        model.logger.info("Running comparisons to baselines.")
        model.compare_to_baselines(train_frac=0.3)
    else:   
        fit, df = model.fit(CYCLES=1, WARMUP=10000, SAMPLES=10000)
        model.plot_histogram(fit, df)
        model.plot_scatter(fit, df)
        model.plot_results(fit, df)
        model.write_estimate(fit, df)
        model.logger.info(f"Generating maps for {model.RUNID}")
        generate_maps(model.RUNID, f"runs/{model.RUNID}/estimate_at_least_one_positive_image_by_area.csv", estimate='at_least_one_positive_image_by_area')
        generate_maps(model.RUNID, f"runs/{model.RUNID}/estimate_p_y.csv", estimate='p_y')
        
        model.logger.info(f"Generating NYC analysis dataframe for {model.RUNID}")
        generate_nyc_analysis_df(run_dir=f"runs/{model.RUNID}", custom_prefix=args.prefix, use_smoothing=True, logger=model.logger)

        model.logger.success("All items in main program routine completed.")


