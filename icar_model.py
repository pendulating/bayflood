# Dashcam-Bayesian Flooding Model Project 
# Developers: Matthew Franchi (mattwfranchi) and Emma Pierson (epierson9)
# Cornell Tech 

# In this script, we house a class that fits various Stan models to a processed dataset of urban street flooding conditions in New York City. 


## Module Imports 
import util

import json
import datetime
import os
import logger
import multiprocessing
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("fork")

import pandas as pd
import stan
import numpy as np
from scipy.stats import pearsonr
import arviz as az
from scipy.special import expit
import matplotlib.pyplot as plt
import nest_asyncio

import sys

import warnings
import argparse

from generate_maps import generate_maps
from refresh_cache import refresh_cache

LATEX_PLOTTING=False
if LATEX_PLOTTING:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

nest_asyncio.apply()


## Class Definition
class ICAR_MODEL:
    def __init__(
        self,
        ICAR_PRIOR_SETTING="none",
        ANNOTATIONS_HAVE_LOCATIONS=True,
        SIMULATED_DATA=False,
        ESTIMATE_PARAMS=[],
        EMPIRICAL_DATA_PATH="",
        adj=[],
        adj_matrix_storage=None
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
        self.use_external_covariates = False
        self.EMPIRICAL_DATA_PATH = EMPIRICAL_DATA_PATH

        self.icar_prior_setting = ICAR_PRIOR_SETTING
        assert self.icar_prior_setting in [
            "none",
            "cheating",
            "proper",
            "just_model_p_y",
        ]

        self.VALID_ESTIMATE_PARAMETERS = ["p_y", "at_least_one_positive_image_by_area"]
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
            "weighted_ICAR_prior_annotations_have_locations": open(
                "stan_models/weighted_ICAR_prior_annotations_have_locations.stan"
            ).read(),
            "weighted_ICAR_prior_annotations_have_locations_external_covariates": open(
                "stan_models/weighted_ICAR_prior_annotations_have_locations_external_covariates.stan"
            ).read(),
        }

        self.logger = logger.setup_logger(f"ICAR_MODEL: {ICAR_PRIOR_SETTING}, ahl {ANNOTATIONS_HAVE_LOCATIONS}, simulated {SIMULATED_DATA}")
        self.logger.setLevel("INFO")
        self.logger.info("ICAR_MODEL instance initialized.")

        self.adj_path = adj
        self.adj_matrix_storage = adj_matrix_storage

        self.RUNID = "NOT_SET"


        # other misc sanity checks 

        # cannot use the at_least_one_positive_image_by_area parameter if additional annotation location data is not utilized 
        if not self.annotations_have_locations: 
            assert 'at_least_one_positive_image_by_area' not in self.ESTIMATE_PARAMETERS

        # refresh cache 
        

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

            self.logger.success("Successfully generated simulated data.")
        else:
            self.logger.info("Reading empirical data.")
            self.data_to_use = util.read_real_data(fpath=self.EMPIRICAL_DATA_PATH,
                annotations_have_locations=self.annotations_have_locations, adj=self.adj_path, adj_matrix_storage=self.adj_matrix_storage, 
                                use_external_covariates = self.use_external_covariates
            )
            self.logger.success("Successfully read empirical data.")

            # validate observed data
            observed_data_copy = self.parse_data_for_validation()
            util.validate_observed_data(
                observed_data_copy, self.annotations_have_locations
            )
            self.logger.success("Successfully validated the observed data.")
            del observed_data_copy



    def fit(self, CYCLES=1, WARMUP=1000, SAMPLES=1500):
        self.RUNID = datetime.datetime.now().strftime("%Y%m%d-%H%M")

        # add parent dirs that split runs based on simulated or empirical, annotations_have_locations, and icar_prior_setting
        self.RUNID = f"icar_{self.icar_prior_setting}/simulated_{self.use_simulated_data}/ahl_{self.annotations_have_locations}/{self.RUNID}"

        os.makedirs(f"runs/{self.RUNID}", exist_ok=True)

        for i in range(CYCLES):
            self.load_data()

            if self.icar_prior_setting == "cheating":
                self.logger.info("Building model with cheating ICAR prior.")
                self.data_to_use["observed_data"]["use_ICAR_prior"] = 1
                self.data_to_use["observed_data"]["ICAR_prior_weight"] = 0.5
                if self.annotations_have_locations:
                    self.logger.info(
                        "Building model with annotations have locations."
                    )
                    if not self.use_external_covariates:
                        model = stan.build(
                            self.models["weighted_ICAR_prior_annotations_have_locations"],
                            data=self.data_to_use["observed_data"],
                        )
                    else:
                        model = stan.build(self.models['weighted_ICAR_prior_annotations_have_locations_external_covariates'], 
                            data=self.data_to_use['observed_data'])
                        self.ADDITIONAL_PARAMS_TO_SAVE += ['external_covariate_slopes', 'external_covariate_intercepts']

                else:
                    self.logger.info("Building model without annotation location data.")
                    model = stan.build(
                        self.models["weighted_ICAR_prior"],
                        data=self.data_to_use["observed_data"],
                    )
            elif self.icar_prior_setting == "none":
                self.logger.info("Building model with no ICAR prior.")
                self.data_to_use["observed_data"]["use_ICAR_prior"] = 0
                self.data_to_use["observed_data"]["ICAR_prior_weight"] = 0
                if self.annotations_have_locations:
                    self.logger.info(
                        "Building model with annotations have locations."
                    )
                    if not self.use_external_covariates:
                        model = stan.build(
                            self.models["weighted_ICAR_prior_annotations_have_locations"],
                            data=self.data_to_use["observed_data"],
                        )
                    else:
                        model = stan.build(self.models['weighted_ICAR_prior_annotations_have_locations_external_covariates'], 
                            data=self.data_to_use['observed_data'])
                        self.ADDITIONAL_PARAMS_TO_SAVE += ['external_covariate_slopes', 'external_covariate_intercepts']

                    
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


            self.logger.info(f"Successfully built the model, with use_icar_prior: {self.data_to_use['observed_data']['use_ICAR_prior']} and icar_prior_weight: {self.data_to_use['observed_data']['ICAR_prior_weight']}.")

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
                "CYCLES": CYCLES,
                "WARMUP": WARMUP,
                "SAMPLES": SAMPLES,
                "use_icar_prior": self.data_to_use["observed_data"]["use_ICAR_prior"],
                "icar_prior_weight": self.data_to_use["observed_data"]["ICAR_prior_weight"],
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

    def plot_results(self, fit, df):
        if self.icar_prior_setting == "just_model_p_y":
            print(
                az.summary(
                    fit,
                    var_names=[
                        "p_y_1_given_y_hat_1",
                        "p_y_1_given_y_hat_0",
                        "p_y_hat_1_given_y_1",
                        "p_y_hat_1_given_y_0",
                        "empirical_p_yhat",
                    ] + self.ESTIMATE_PARAMETERS + self.ADDITIONAL_PARAMS_TO_SAVE,
                )
            )

            # also, write to file 
            with open(f"runs/{self.RUNID}/summary.txt", "w") as f:
                f.write(
                    az.summary(
                        fit,
                        var_names=[
                            "p_y_hat_1_given_y_1",
                            "p_y_hat_1_given_y_0",
                            "phi_offset",
                            "p_y_1_given_y_hat_1",
                            "p_y_1_given_y_hat_0",
                            "empirical_p_yhat",
                        ] + self.ESTIMATE_PARAMETERS + self.ADDITIONAL_PARAMS_TO_SAVE,
                    ).to_string()
                )

        else:
            print(
                az.summary(
                    fit,
                    var_names=[
                        "p_y_hat_1_given_y_1",
                        "p_y_hat_1_given_y_0",
                        "phi_offset",
                        "p_y_1_given_y_hat_1",
                        "p_y_1_given_y_hat_0",
                        "empirical_p_yhat",
                    ] + self.ESTIMATE_PARAMETERS + self.ADDITIONAL_PARAMS_TO_SAVE,
                )
            )

            # also, write to file 
            with open(f"runs/{self.RUNID}/summary.txt", "w") as f:
                f.write(
                    az.summary(
                        fit,
                        var_names=[
                            "p_y_hat_1_given_y_1",
                            "p_y_hat_1_given_y_0",
                            "phi_offset",
                            "p_y_1_given_y_hat_1",
                            "p_y_1_given_y_hat_0",
                            "empirical_p_yhat",
                        ] + self.ESTIMATE_PARAMETERS + self.ADDITIONAL_PARAMS_TO_SAVE,
                    ).to_string()
                )

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
                    prior_on_p_y = expit(df["phi_offset"]).mean()
                    plt.axhline(expit(prior_on_p_y), color="black", linestyle="--")
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

    # Add arguments
    parser.add_argument(
        "icar_prior_setting", 
        type=str, 
        help="The setting for the ICAR prior."
    )

    parser.add_argument(
        "annotations_have_locations", 
        type=lambda x: x in ['True', 'False'],
        choices=['True', 'False'],
        help="Whether annotations have locations (True/False)."
    )

    parser.add_argument(
        "simulated_data", 
        type=lambda x: x in ['True', 'False'],
        choices=['True', 'False'],
        help="Whether the data is simulated (True/False)."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    icar_prior_setting = args.icar_prior_setting
    annotations_have_locations = args.annotations_have_locations == 'True'
    simulated_data = args.simulated_data == 'True'


    model = ICAR_MODEL(
        ICAR_PRIOR_SETTING=icar_prior_setting,
        ESTIMATE_PARAMS=["p_y", "at_least_one_positive_image_by_area"],
        ANNOTATIONS_HAVE_LOCATIONS=annotations_have_locations,
        SIMULATED_DATA=simulated_data,
        EMPIRICAL_DATA_PATH="data/processed/flooding_ct_dataset.csv",
        adj=["data/processed/ct_nyc_adj_list_node1.txt","data/processed/ct_nyc_adj_list_node2.txt"],
        adj_matrix_storage=False
    )
    #
    fit, df = model.fit(CYCLES=1, WARMUP=5000, SAMPLES=1000)
    model.plot_histogram(fit, df)
    model.plot_scatter(fit, df)
    model.plot_results(fit, df)
    model.write_estimate(fit, df)
    model.logger.info(f"Generating maps for {model.RUNID}")
    generate_maps(model.RUNID, f"runs/{model.RUNID}/estimate_at_least_one_positive_image_by_area.csv", estimate='at_least_one_positive_image_by_area')
    generate_maps(model.RUNID, f"runs/{model.RUNID}/estimate_p_y.csv", estimate='p_y')
    model.logger.success("All items in main program routine completed.")
