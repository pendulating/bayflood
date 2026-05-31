# In this script, we house a class that fits various Stan models to a processed dataset of urban street flooding conditions in New York City. 

# Set cache directory BEFORE importing stan/httpstan
# This avoids disk quota issues on home directory
import os
from pathlib import Path
_project_root = Path(__file__).parent.resolve()
_cache_dir = _project_root / ".cache"
os.environ["XDG_CACHE_HOME"] = str(_cache_dir)

## Module Imports 
import util
import config
from geometry_config import GeometryType, get_geometry_config
from IPython import embed


import json
from copy import deepcopy
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import datetime
import logger
import multiprocessing
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("fork")

# Patch aiohttp timeout for large models (CBG has ~6800 areas)
# Default timeout is too short to transfer large fit results
import aiohttp
_original_client_session_init = aiohttp.ClientSession.__init__
def _patched_client_session_init(self, *args, **kwargs):
    if 'timeout' not in kwargs:
        # 30 minute total timeout for large models
        kwargs['timeout'] = aiohttp.ClientTimeout(total=1800)
    _original_client_session_init(self, *args, **kwargs)
aiohttp.ClientSession.__init__ = _patched_client_session_init

import pandas as pd
import stan as stan
import numpy as np
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
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
    """
    Intrinsic Conditional Autoregressive (ICAR) model for urban street flooding analysis.
    
    This class implements Bayesian spatial modeling for flood detection using dashcam imagery.
    It supports multiple prior specifications, external covariates, and various estimation
    parameters for comprehensive flooding analysis.
    
    Attributes:
        N_ANNOTATED_CLASSIFIED_NEGATIVE (int): Number of annotated negative samples
        N_ANNOTATED_CLASSIFIED_POSITIVE (int): Number of annotated positive samples
        N_SIMULATED_TRACTS (int): Number of tracts for simulated data
        annotations_have_locations (bool): Whether annotations include spatial locations
        use_simulated_data (bool): Whether to use simulated or empirical data
        use_external_covariates (bool): Whether to include external covariates
        icar_prior_setting (str): Type of spatial prior ("none", "icar", "proper", "just_model_p_y")
        ESTIMATE_PARAMETERS (list): Parameters to estimate from the model
        models (dict): Dictionary of Stan model specifications
        logger: Logger instance for tracking progress
        
    Example:
        >>> model = ICAR_MODEL(
        ...     PREFIX='flooding_analysis',
        ...     ICAR_PRIOR_SETTING="icar",
        ...     ANNOTATIONS_HAVE_LOCATIONS=True,
        ...     EXTERNAL_COVARIATES=True,
        ...     ESTIMATE_PARAMS=['p_y', 'at_least_one_positive_image_by_area'],
        ...     EMPIRICAL_DATA_PATH="data/processed/flooding_ct_dataset.csv",  # or flooding_cbg_dataset.csv
        ...     geometry_type="ct"  # or "cbg", "cb"
        ... )
        >>> model.load_data()
        >>> fit = model.fit(CYCLES=1, WARMUP=1000, SAMPLES=1500)
    """
    
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
        downsample_frac=1,
        DOWNSAMPLE_ALL_IMAGES=False,
        downsample_seed=None,
        trim_to_median: bool = False,
        trim_remove_frac: float | None = None,
        USE_CATCH_BASINS=False,
        geometry_type: str | GeometryType = "ct",
    ):

        refresh_cache()
        print(SIMULATED_DATA)
        
        # Handle geometry type
        if isinstance(geometry_type, str):
            geometry_type = GeometryType(geometry_type.lower())
        self.geometry_type = geometry_type
        self.geometry_config = get_geometry_config(geometry_type)
        self.id_column = self.geometry_config.id_column

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
        self.use_catch_basins = USE_CATCH_BASINS
        self.downsample_frac = downsample_frac
        self.downsample_all_images = DOWNSAMPLE_ALL_IMAGES
        self.downsample_seed = downsample_seed
        self.trim_to_median = trim_to_median
        self.trim_remove_frac = trim_remove_frac
        self.trim_history = []
        self.EMPIRICAL_DATA_PATH = EMPIRICAL_DATA_PATH

        self.icar_prior_setting = ICAR_PRIOR_SETTING
        assert self.icar_prior_setting in ["icar"], "Only 'icar' setting is supported in this artifact."

        self.VALID_ESTIMATE_PARAMETERS = ["p_y", "at_least_one_positive_image_by_area", "at_least_one_positive_image_by_area_if_you_have_100_images"]
        self.ADDITIONAL_PARAMS_TO_SAVE = []
        self.ESTIMATE_PARAMETERS = ESTIMATE_PARAMS
        for p in self.ESTIMATE_PARAMETERS:
            assert p in self.VALID_ESTIMATE_PARAMETERS


        # This dictionary stores the available stan models
        self.models = {
            "ICAR_prior_annotations_have_locations": open(
                "stan_models/ICAR_prior_annotations_have_locations.stan"
            ).read(),
            "weighted_ICAR_prior": open(
                "stan_models/weighted_ICAR_prior.stan"
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
        """
        Parse and prepare observed data for validation and debugging.
        
        Converts numpy arrays to lists and handles int64 serialization issues
        that can occur when saving data to JSON format.
        
        Returns:
            dict: Copy of observed data with numpy arrays converted to lists
                 and int64 values converted to regular integers
        """
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
        """
        Load and prepare data for ICAR model fitting.
        
        Depending on configuration, either generates simulated data or loads
        empirical data from file. Handles data validation, downsampling, and
        external covariates processing.
        
        The loaded data is stored in self.data_to_use and contains:
        - observed_data: Dictionary with Stan model inputs
        - external covariates (if enabled)
        - adjacency information (if provided)
        
        Raises:
            FileNotFoundError: If empirical data file is not found
            ValueError: If data validation fails
        """
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
                mode = "all images" if self.downsample_all_images else "annotated images"
                self.logger.info(f"Downsampling {mode} with downsample_frac={self.downsample_frac}.")
                self.data_to_use = self.downsample_data(
                    self.data_to_use,
                    downsample_frac=self.downsample_frac,
                    downsample_all_images=self.downsample_all_images,
                    seed=self.downsample_seed,
                )

            self.logger.success("Successfully generated simulated data.")
        else:
            self.logger.info("Reading empirical data.")
            self.data_to_use, external_covariates_info = util.read_real_data(
                fpath=self.EMPIRICAL_DATA_PATH,
                annotations_have_locations=self.annotations_have_locations, 
                adj=self.adj_path, 
                adj_matrix_storage=self.adj_matrix_storage, 
                use_external_covariates=self.use_external_covariates,
                use_catch_basins=self.use_catch_basins,
                id_column=self.id_column
            )

            if self.use_external_covariates:
                # write external covariates to file for debugging
                print(external_covariates_info)
                external_covariates_info = pd.DataFrame.from_dict(external_covariates_info['external_covariates'])
                with open(f"runs/{self.RUNID}/external_covariates.csv", "w") as f:
                    external_covariates_info.to_csv(f)
                    
            self.logger.success("Successfully read empirical data.")

            if self.trim_to_median:
                self.logger.info(
                    "Applying trim_to_median (remove_frac=%s)."
                    % (self.trim_remove_frac if self.trim_remove_frac is not None else (1.0 - float(self.downsample_frac)))
                )
                self.data_to_use = self.iterative_trim_to_median(
                    self.data_to_use,
                    remove_frac=self.trim_remove_frac,
                )
                self.logger.success("Successfully applied trim_to_median.")

            if (not self.trim_to_median) and (self.downsample_frac < 1):
                mode = "all images" if self.downsample_all_images else "annotated images"
                self.logger.info(f"Downsampling {mode} with downsample_frac={self.downsample_frac}.")
                self.data_to_use = self.downsample_data(
                    self.data_to_use,
                    downsample_frac=self.downsample_frac,
                    downsample_all_images=self.downsample_all_images,
                    seed=self.downsample_seed,
                )

            # validate observed data
            if self.trim_to_median:
                # trim_to_median intentionally changes annotated totals, which breaks the strict validation
                self.logger.info("Skipping strict validate_observed_data (trim_to_median enabled).")
            else:
                observed_data_copy = self.parse_data_for_validation()
                util.validate_observed_data(
                    observed_data_copy, self.annotations_have_locations, self.downsample_frac
                )
                self.logger.success("Successfully validated the observed data.")
                del observed_data_copy



    def fit(self, CYCLES=1, WARMUP=1000, SAMPLES=1500, data_already_loaded=False):
        # pass in data_already_loaded = True if you want to use data that's already been loaded in. 
        # by default the method reloads the data. 
        if not data_already_loaded:
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
                    
                    model_name = "ICAR_prior_annotations_have_locations"
                    self.logger.info(f"Using model specification: {model_name}")

                    model = stan.build(
                        self.models[model_name],
                        data=self.data_to_use["observed_data"],
                    )
                    self.ADDITIONAL_PARAMS_TO_SAVE += ['spatial_sigma', 'external_covariate_beta']
                else:
                    raise ValueError("This artifact requires annotations_have_locations=True.")


            self.logger.info(f"Successfully built the model, with use_icar_prior: {self.data_to_use['observed_data']['use_ICAR_prior']}.")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
                "USE_CATCH_BASINS": self.use_catch_basins,
                "CYCLES": CYCLES,
                "WARMUP": WARMUP,
                "SAMPLES": SAMPLES,
                "DOWNSAMPLE_ALL_IMAGES": self.downsample_all_images,
                "downsample_frac": self.downsample_frac,
                "downsample_seed": self.downsample_seed,
                "trim_to_median": self.trim_to_median,
                "trim_remove_frac": None if self.trim_remove_frac is None else float(self.trim_remove_frac),
                "trim_history": self.trim_history,
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
            if k in ['N', 'N_edges', 'node1', 'node2', 'tract_id', 'geoid', 'center_of_phi_offset_prior', 'external_covariates', 'n_external_covariates']:
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
            if k not in ['N', 'N_edges', 'node1', 'node2', 'tract_id', 'geoid', 'center_of_phi_offset_prior', 'external_covariates', 'n_external_covariates']:
                assert (train_data[k] + test_data[k] == full_dataset['observed_data'][k]).all()
        print("With a train frac of %2.3f, train set has %i total images; test set has %i" % 
                (train_frac, train_data['n_images_by_area'].sum(), test_data['n_images_by_area'].sum()))
        return train_data, test_data

    def downsample_data(self, full_dataset, downsample_frac=0.1, downsample_all_images=False, seed=None):
        """
        Downsample the dataset by downsample_frac.
        If downsample_all_images is False, only annotated images are downsampled.
        If True, all base count fields (annotated and non-annotated) are downsampled
        before recomputing derived totals.

        ``seed`` makes the random binomial thinning reproducible. When None, the
        draw is non-deterministic (preserving prior behavior).
        """
        rng = np.random.default_rng(seed)
        if seed is not None:
            self.logger.info(f"Downsampling with fixed seed={seed}.")
        downsampled_data = deepcopy(full_dataset)
        observed = downsampled_data['observed_data']
        original_observed = full_dataset['observed_data']

        if downsample_all_images:
            # Ensure we have both positive and negative non-annotated counts
            if 'n_non_annotated_by_area_classified_negative' not in original_observed:
                inferred_negative = (
                    original_observed['n_non_annotated_by_area']
                    - original_observed['n_non_annotated_by_area_classified_positive']
                )
                assert (inferred_negative >= 0).all()
                original_observed['n_non_annotated_by_area_classified_negative'] = inferred_negative

            base_fields = [
                'n_classified_positive_annotated_positive_by_area',
                'n_classified_positive_annotated_negative_by_area',
                'n_classified_negative_annotated_negative_by_area',
                'n_classified_negative_annotated_positive_by_area',
                'n_non_annotated_by_area_classified_positive',
                'n_non_annotated_by_area_classified_negative',
            ]

            for k in base_fields:
                observed[k] = rng.binomial(original_observed[k], downsample_frac)
                assert (observed[k] >= 0).all()

            observed['n_annotated_by_area'] = (
                observed['n_classified_positive_annotated_positive_by_area'] +
                observed['n_classified_positive_annotated_negative_by_area'] +
                observed['n_classified_negative_annotated_negative_by_area'] +
                observed['n_classified_negative_annotated_positive_by_area']
            )
            observed['n_non_annotated_by_area'] = (
                observed['n_non_annotated_by_area_classified_positive'] +
                observed['n_non_annotated_by_area_classified_negative']
            )
            observed['n_images_by_area'] = observed['n_non_annotated_by_area'] + observed['n_annotated_by_area']
            observed['n_classified_positive_by_area'] = (
                observed['n_classified_positive_annotated_positive_by_area'] +
                observed['n_classified_positive_annotated_negative_by_area'] +
                observed['n_non_annotated_by_area_classified_positive']
            )
            observed['total_annotated_classified_positive'] = (
                observed['n_classified_positive_annotated_positive_by_area'] +
                observed['n_classified_positive_annotated_negative_by_area']
            )
            observed['total_annotated_classified_negative'] = (
                observed['n_classified_negative_annotated_positive_by_area'] +
                observed['n_classified_negative_annotated_negative_by_area']
            )

            self.logger.info(
                f"Original total images sum: {original_observed['n_images_by_area'].sum()}; "
                f"downsampled total images sum: {observed['n_images_by_area'].sum()}"
            )
        else:
            original_observed['n_annotated_by_area'] = (
                original_observed['n_classified_positive_annotated_positive_by_area'] +
                original_observed['n_classified_positive_annotated_negative_by_area'] +
                original_observed['n_classified_negative_annotated_negative_by_area'] +
                original_observed['n_classified_negative_annotated_positive_by_area']
            )
            
            annotated_fields = [
                'n_classified_positive_annotated_positive_by_area',
                'n_classified_positive_annotated_negative_by_area',
                'n_classified_negative_annotated_negative_by_area',
                'n_classified_negative_annotated_positive_by_area'
            ]
            
            for k in annotated_fields:
                observed[k] = rng.binomial(
                    original_observed[k],
                    downsample_frac
                )
                assert (observed[k] >= 0).all()
            
            observed['n_annotated_by_area'] = (
                observed['n_classified_positive_annotated_positive_by_area'] + 
                observed['n_classified_positive_annotated_negative_by_area'] + 
                observed['n_classified_negative_annotated_negative_by_area'] + 
                observed['n_classified_negative_annotated_positive_by_area']
            )
            
            observed['n_images_by_area'] = (
                observed['n_non_annotated_by_area'] +
                observed['n_annotated_by_area']
            )
            
            observed['n_classified_positive_by_area'] = (
                observed['n_classified_positive_annotated_positive_by_area'] +
                observed['n_classified_positive_annotated_negative_by_area'] +
                observed['n_non_annotated_by_area_classified_positive']
            )

            observed['total_annotated_classified_positive'] = (
                observed['n_classified_positive_annotated_positive_by_area'] +
                observed['n_classified_positive_annotated_negative_by_area']
            )

            observed['total_annotated_classified_negative'] = (
                observed['n_classified_negative_annotated_positive_by_area'] +
                observed['n_classified_negative_annotated_negative_by_area']
            )

            self.logger.info(f"Original annotated images: {original_observed['n_annotated_by_area'].sum()}")
            self.logger.info(f"Downsampled annotated images: {observed['n_annotated_by_area'].sum()}")
            self.logger.info(f"Total images after downsampling: {observed['n_images_by_area'].sum()}")
        
        return downsampled_data

    def iterative_trim_to_median(
        self,
        full_dataset,
        remove_frac: float | None = None,
    ):
        """
        Trim high-count tracts toward the current median count to fill a global removal budget.

        This operates purely on per-tract count fields (no per-image sampling available).
        We compute a global removal budget (\"moat\") as remove_frac * total_images,
        and iteratively remove counts from high-count tracts until the budget is filled.

        Each pass:
        - Compute median_i (integer floor) from current n_images_by_area (pre-trim)
        - Consider tracts with Ci > median_i, with per-tract capacity cap_i = Ci - median_i
        - Allocate the pass removal target across high tracts proportionally to cap_i,
          respecting cap_i (so we don't drop below the current median within a pass)
        - For each tract, allocate removals across 6 base fields (without replacement):
            - 4 annotated fields
            - non-annotated classified positive
            - non-annotated classified negative (inferred as n_non_annotated - n_non_annotated_pos)
        - Recompute derived totals and record Wasserstein/EMD vs constant-at-median_i target.
        """
        if not self.annotations_have_locations:
            raise ValueError("iterative_trim_to_median requires annotations_have_locations=True.")

        # Default: if remove_frac not provided, use trim_remove_frac, else fall back to (1 - downsample_frac)
        if remove_frac is None:
            if self.trim_remove_frac is not None:
                remove_frac = self.trim_remove_frac
            else:
                remove_frac = 1.0 - float(self.downsample_frac)

        if remove_frac <= 0 or remove_frac >= 1:
            raise ValueError(f"remove_frac must be in (0, 1); got {remove_frac}.")

        trimmed = deepcopy(full_dataset)
        observed = trimmed["observed_data"]

        required = [
            "n_images_by_area",
            "n_classified_positive_by_area",
            "n_classified_positive_annotated_positive_by_area",
            "n_classified_positive_annotated_negative_by_area",
            "n_classified_negative_annotated_negative_by_area",
            "n_classified_negative_annotated_positive_by_area",
            "n_non_annotated_by_area",
            "n_non_annotated_by_area_classified_positive",
        ]
        missing = [k for k in required if k not in observed]
        if missing:
            raise ValueError(f"Missing required observed_data keys for trim_to_median: {missing}")

        def _as_int_array(x):
            arr = np.asarray(x)
            if np.any(arr < 0):
                raise ValueError("Negative counts encountered in observed_data before trimming.")
            return arr.astype(np.int64, copy=True)

        # Pull out base fields as int arrays we will mutate.
        ann_pp = _as_int_array(observed["n_classified_positive_annotated_positive_by_area"])
        ann_pn = _as_int_array(observed["n_classified_positive_annotated_negative_by_area"])
        ann_nn = _as_int_array(observed["n_classified_negative_annotated_negative_by_area"])
        ann_np = _as_int_array(observed["n_classified_negative_annotated_positive_by_area"])
        non_total = _as_int_array(observed["n_non_annotated_by_area"])
        non_pos = _as_int_array(observed["n_non_annotated_by_area_classified_positive"])

        if np.any(non_pos > non_total):
            raise ValueError("Found n_non_annotated_by_area_classified_positive > n_non_annotated_by_area.")

        def _recompute_derived_fields():
            observed["n_classified_positive_annotated_positive_by_area"] = ann_pp
            observed["n_classified_positive_annotated_negative_by_area"] = ann_pn
            observed["n_classified_negative_annotated_negative_by_area"] = ann_nn
            observed["n_classified_negative_annotated_positive_by_area"] = ann_np
            observed["n_non_annotated_by_area"] = non_total
            observed["n_non_annotated_by_area_classified_positive"] = non_pos

            observed["n_annotated_by_area"] = ann_pp + ann_pn + ann_nn + ann_np
            observed["n_images_by_area"] = observed["n_annotated_by_area"] + non_total
            observed["n_classified_positive_by_area"] = ann_pp + ann_pn + non_pos

            # Keep parity with existing downsampling code, even though names are confusing.
            observed["total_annotated_classified_positive"] = ann_pp + ann_pn
            observed["total_annotated_classified_negative"] = ann_np + ann_nn

            # Lightweight consistency checks
            if np.any(observed["n_images_by_area"] < 0):
                raise ValueError("Negative n_images_by_area after trimming.")
            if np.any(observed["n_classified_positive_by_area"] < 0):
                raise ValueError("Negative n_classified_positive_by_area after trimming.")
            if np.any(non_total < 0) or np.any(non_pos < 0):
                raise ValueError("Negative non-annotated counts after trimming.")
            if np.any(non_pos > non_total):
                raise ValueError("non_pos exceeded non_total after trimming.")

        def _sample_multivariate_hypergeometric(counts6: np.ndarray, nremove: int) -> np.ndarray:
            """
            Sample removal counts across categories *without replacement*.
            Uses sequential hypergeometric draws (exact for multivariate hypergeometric).
            """
            counts6 = counts6.astype(np.int64, copy=False)
            if nremove <= 0:
                return np.zeros_like(counts6)
            total = int(counts6.sum())
            nremove = min(int(nremove), total)
            removed = np.zeros_like(counts6)
            remaining_total = total
            remaining_to_remove = nremove

            for j in range(len(counts6) - 1):
                if remaining_to_remove <= 0:
                    break
                ngood = int(counts6[j])
                if ngood <= 0:
                    remaining_total -= ngood
                    continue
                nbad = remaining_total - ngood
                draw = int(np.random.hypergeometric(ngood, nbad, remaining_to_remove))
                draw = min(draw, ngood, remaining_to_remove)
                removed[j] = draw
                remaining_to_remove -= draw
                remaining_total -= ngood

            removed[-1] = remaining_to_remove
            if removed[-1] > counts6[-1]:
                raise ValueError("Internal sampling error: removal exceeded available count in last category.")
            return removed

        self.trim_history = []
        _recompute_derived_fields()

        total_images_start = int(_as_int_array(observed["n_images_by_area"]).sum())
        removal_budget_target = int(np.ceil(float(remove_frac) * total_images_start))
        remaining_budget = removal_budget_target

        self.logger.info(
            "trim_to_median: starting moat fill remove_frac=%.4f total_images=%d budget=%d"
            % (float(remove_frac), total_images_start, removal_budget_target)
        )

        p = 0
        safety_cap = 10_000  # hard safety cap to prevent infinite loops in case of unexpected data/pathology
        while remaining_budget > 0:
            if p >= safety_cap:
                raise ValueError(f"trim_to_median: exceeded safety cap of {safety_cap} passes.")
            if remaining_budget <= 0:
                break

            C_before = _as_int_array(observed["n_images_by_area"])
            mean_before = float(C_before.mean())
            median_target = int(np.median(C_before))  # integer floor if even N

            cap = C_before - median_target
            cap[cap < 0] = 0
            sum_cap = int(cap.sum())
            if sum_cap <= 0:
                # Degenerate case: no tracts above median (e.g., uniform counts). To still fill the moat,
                # allow removal from all tracts proportionally to their current counts.
                cap = C_before.copy()
                cap[cap < 0] = 0
                sum_cap = int(cap.sum())
                if sum_cap <= 0:
                    raise ValueError("trim_to_median: cannot fill moat (no images left to remove).")

            pass_target = min(int(remaining_budget), sum_cap)
            if pass_target <= 0:
                break

            # Allocate removals across tracts proportionally to cap, respecting per-tract caps.
            desired = pass_target * (cap.astype(float) / float(sum_cap))
            tract_remove = np.floor(desired).astype(np.int64)
            tract_remove = np.minimum(tract_remove, cap.astype(np.int64))

            remainder = int(pass_target - int(tract_remove.sum()))
            if remainder > 0:
                frac = desired - tract_remove.astype(float)
                # distribute leftover 1s to largest fractional parts first, respecting cap
                order = np.argsort(-frac)
                for t in order:
                    if remainder <= 0:
                        break
                    if cap[t] <= tract_remove[t]:
                        continue
                    tract_remove[t] += 1
                    remainder -= 1
                # if still remainder (due to caps), distribute to any tract with remaining cap
                if remainder > 0:
                    avail = np.where(cap > tract_remove)[0]
                    for t in avail:
                        if remainder <= 0:
                            break
                        add = int(min(remainder, int(cap[t] - tract_remove[t])))
                        tract_remove[t] += add
                        remainder -= add
            if int(tract_remove.sum()) != pass_target:
                raise ValueError("Failed to allocate pass removal target across tracts.")

            total_removed = 0
            n_high = int((cap > 0).sum())
            idxs = np.where(tract_remove > 0)[0]
            for t in idxs:
                nremove = int(tract_remove[t])

                non_neg_t = int(non_total[t] - non_pos[t])
                if non_neg_t < 0:
                    raise ValueError("Negative inferred non-annotated classified negative count.")

                counts6 = np.array(
                    [
                        int(ann_pp[t]),
                        int(ann_pn[t]),
                        int(ann_nn[t]),
                        int(ann_np[t]),
                        int(non_pos[t]),
                        int(non_neg_t),
                    ],
                    dtype=np.int64,
                )
                tract_total = int(counts6.sum())
                if tract_total <= 0:
                    continue
                nremove = min(nremove, tract_total)

                removed6 = _sample_multivariate_hypergeometric(counts6, nremove)
                if removed6.sum() != nremove:
                    raise ValueError("Removal allocation did not sum to requested nremove.")

                ann_pp[t] -= removed6[0]
                ann_pn[t] -= removed6[1]
                ann_nn[t] -= removed6[2]
                ann_np[t] -= removed6[3]
                non_pos[t] -= removed6[4]
                non_total[t] -= (removed6[4] + removed6[5])

                total_removed += int(nremove)

            _recompute_derived_fields()

            C_after = np.asarray(observed["n_images_by_area"], dtype=float)
            mean_after = float(C_after.mean())
            emd = float(wasserstein_distance(C_after, np.full_like(C_after, median_target)))

            self.trim_history.append(
                {
                    "pass": int(p),
                    "mean_before": float(mean_before),
                    "median_target": float(median_target),
                    "mean_after": float(mean_after),
                    "n_high": int(n_high),
                    "total_removed": int(total_removed),
                    "remaining_budget_after": int(max(0, remaining_budget - total_removed)),
                    "emd_to_median_target": float(emd),
                }
            )

            self.logger.info(
                "trim_to_median pass=%d median_target=%d mean_before=%.3f mean_after=%.3f "
                "n_high=%d removed=%d remaining_budget=%d emd=%.6f"
                % (p, median_target, mean_before, mean_after, n_high, total_removed, max(0, remaining_budget - total_removed), emd)
            )

            if total_removed == 0:
                self.logger.info("trim_to_median: stopping (no removals this iteration).")
                break
            remaining_budget -= int(total_removed)
            p += 1

        return trimmed
        
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
        # drop the intercept
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

        self.RUNID = self.RUNID + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M")

        # add parent dirs that split runs based on simulated or empirical, annotations_have_locations, and icar_prior_setting
        self.RUNID = f"icar_{self.icar_prior_setting}/simulated_{self.use_simulated_data}/ahl_{self.annotations_have_locations}/covariates_{self.use_external_covariates}/{self.RUNID}"

        os.makedirs(f"runs/{self.RUNID}", exist_ok=True)

        self.load_data()
        train_data, test_data = self.divide_data_into_train_and_test_set(self.data_to_use, train_frac=train_frac)
        method_and_baselines = self.extract_baselines(train_data)
        ground_truth = self.extract_baselines(test_data)

        self.data_to_use = {'observed_data':train_data}
        fit, df = self.fit(CYCLES=1, WARMUP=12000, SAMPLES=12000, data_already_loaded=True)

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
                    self.logger.error(f"r_hat for parameter {i} is {row['r_hat']}, above threshold of {rhat_thres}")
                    # end the program 
                    raise ValueError(f"r_hat for parameter {i} is {row['r_hat']}, above threshold of {rhat_thres}")
    

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
                    # Convert absolute CI quantiles to error-bar MAGNITUDES
                    # (distance from the point estimate). matplotlib's yerr expects
                    # [lower_err, upper_err] = [estimate - 2.5%, 97.5% - estimate],
                    # not the absolute quantile values.
                    _ci = np.array(estimate_CIs)  # (N, 2): [:,0]=2.5%, [:,1]=97.5%
                    estimate_yerr = np.vstack([estimate - _ci[:, 0], _ci[:, 1] - estimate])
                    n_images_by_area = self.data_to_use["observed_data"]["n_images_by_area"]
                    # make errorbar plot
                    image_cutoff = 100

                    plt.errorbar(
                        empirical_estimate[n_images_by_area >= image_cutoff],
                        estimate[n_images_by_area >= image_cutoff],
                        yerr=estimate_yerr[:, n_images_by_area >= image_cutoff],
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
                        yerr=estimate_yerr[:, n_images_by_area < image_cutoff],
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
            # Use geoid (with tract_id fallback for backward compatibility)
            geoid = self.data_to_use["observed_data"].get("geoid", self.data_to_use["observed_data"].get("tract_id"))

            # make df to write (include both geoid and tract_id for backward compatibility)
            results = pd.DataFrame(
                {
                    "geoid": geoid,
                    "tract_id": geoid,  # backward compatibility alias
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
    parser = argparse.ArgumentParser(description="BayFlood ICAR pipeline training script")

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
        '--no_catch_basins',
        action='store_true',
        default=False,
        help="Exclude catch basin covariates (n_catch_basins, catch_basin_density) from external covariates"
    )

    parser.add_argument(
        '--no-catch-basins',
        action='store_true',
        dest='no_catch_basins_alias',
        help="Alias for --no_catch_basins"
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

    parser.add_argument(
        '--downsample_all_images',
        action='store_true',
        default=False,
        help='If set, apply downsample_frac to all images (annotated and non-annotated)'
    )

    parser.add_argument(
        '--downsample_seed',
        action='store',
        type=int,
        default=None,
        help='Seed for the random binomial downsampling, for reproducible runs. '
             'If unset, downsampling is non-deterministic.'
    )

    # Iterative trim-to-median downsampling variant
    parser.add_argument(
        '--trim_to_median',
        action='store_true',
        default=False,
        help='If set, iteratively trim high-count tracts toward the current median (counts-based).'
    )
    parser.add_argument(
        '--trim_remove_frac',
        action='store',
        type=float,
        default=None,
        help='Global fraction of total images to remove when --trim_to_median is set (e.g. 0.25 removes 25%%). '
             'If not provided, falls back to (1 - --downsample_frac).'
    )

    # Data/config path overrides
    parser.add_argument(
        '--empirical_data_path',
        type=str,
        required=False,
        default=None,
        help='Path to empirical dataset CSV (defaults to EMPIRICAL_DATA_PATH env or config.DATASET_PATH)'
    )

    parser.add_argument(
        '--adj_node1_path',
        type=str,
        required=False,
        default=None,
        help='Path to adjacency edge list node1 file (.txt)'
    )
    parser.add_argument(
        '--adj_node2_path',
        type=str,
        required=False,
        default=None,
        help='Path to adjacency edge list node2 file (.txt)'
    )
    parser.add_argument(
        '--adj_npy_path',
        type=str,
        required=False,
        default=None,
        help='Path to adjacency matrix .npy file (mutually exclusive with node1/node2)'
    )

    parser.add_argument(
        '--geometry_type',
        type=str,
        required=False,
        default='ct',
        choices=['ct', 'cbg', 'cb'],
        help='Census geometry type: ct (tract), cbg (block group), cb (block). Default: ct'
    )

    # Parse the arguments
    args = parser.parse_args()

    # Handle alias
    if hasattr(args, 'no_catch_basins_alias') and args.no_catch_basins_alias:
        args.no_catch_basins = True

    # Get geometry-specific paths based on --geometry_type argument
    from geometry_config import get_geometry_paths
    geo_paths = get_geometry_paths(GeometryType(args.geometry_type))

    # Resolve dataset path (CLI override > geometry config)
    empirical_path = args.empirical_data_path or str(geo_paths.flooding_dataset_path)

    # Resolve adjacency inputs (CLI override > geometry config)
    adj = []
    adj_matrix_storage = False
    if args.adj_npy_path:
        adj = [args.adj_npy_path]
        adj_matrix_storage = True
    elif args.adj_node1_path and args.adj_node2_path:
        adj = [args.adj_node1_path, args.adj_node2_path]
        adj_matrix_storage = False
    else:
        # Use geometry-specific adjacency paths
        adj = [
            str(geo_paths.adjacency_node1_path('custom_geometric')),
            str(geo_paths.adjacency_node2_path('custom_geometric'))
        ]
        adj_matrix_storage = False

    model = ICAR_MODEL(
            PREFIX=args.prefix,
            ICAR_PRIOR_SETTING=args.icar_prior_setting,
            ESTIMATE_PARAMS=["p_y", "at_least_one_positive_image_by_area"],
            ANNOTATIONS_HAVE_LOCATIONS=args.annotations_have_locations,
            EXTERNAL_COVARIATES=args.external_covariates if args.external_covariates is not None else config.EXTERNAL_COVARIATES,
            SIMULATED_DATA=args.simulated_data,
            EMPIRICAL_DATA_PATH=empirical_path,
            adj=adj,
            adj_matrix_storage=adj_matrix_storage,
            downsample_frac=args.downsample_frac,
            DOWNSAMPLE_ALL_IMAGES=args.downsample_all_images,
            downsample_seed=args.downsample_seed,
            trim_to_median=args.trim_to_median,
            trim_remove_frac=args.trim_remove_frac,
            USE_CATCH_BASINS=not args.no_catch_basins,
            geometry_type=args.geometry_type,
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
        generate_maps(model.RUNID, f"runs/{model.RUNID}/estimate_at_least_one_positive_image_by_area.csv", estimate='at_least_one_positive_image_by_area', geometry_type=model.geometry_type)
        generate_maps(model.RUNID, f"runs/{model.RUNID}/estimate_p_y.csv", estimate='p_y', geometry_type=model.geometry_type)
        
        model.logger.info(f"Generating NYC analysis dataframe for {model.RUNID}")
        generate_nyc_analysis_df(
            run_dir=f"runs/{model.RUNID}", 
            custom_prefix=args.prefix, 
            logger=model.logger,
            geometry_type=model.geometry_type
        )

        model.logger.success("All items in main program routine completed.")


