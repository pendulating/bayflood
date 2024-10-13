import util
import nest_asyncio

nest_asyncio.apply()
import multiprocessing

# set start method if not set
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("fork")
import pandas as pd
import stan
import numpy as np
from scipy.stats import pearsonr

from scipy.special import expit
import matplotlib.pyplot as plt
# ENABLE LATEX PLOTTING 
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

import json
import arviz as az

import datetime
import os

import logger


class ICAR_MODEL:
    def __init__(
        self,
        ICAR_PRIOR_SETTING="none",
        ANNOTATIONS_HAVE_LOCATIONS=True,
        SIMULATED_DATA=False,
        adj=[],
        adj_matrix_storage=False
    ):
        self.icar_prior_setting = ICAR_PRIOR_SETTING
        assert self.icar_prior_setting in [
            "none",
            "cheating",
            "proper",
            "just_model_p_y",
        ]
        self.N_ANNOTATED_CLASSIFIED_NEGATIVE = 500
        self.N_ANNOTATED_CLASSIFIED_POSITIVE = 500
        self.N_ANNOTATED_CLASSIFIED_NEGATIVE_TRUE_POSITIVE = 3
        self.N_ANNOTATED_CLASSIFIED_POSITIVE_TRUE_POSITIVE = 329
        self.TOTAL_PRED_POSITIVE = 1465
        self.TOTAL_PRED_NEGATIVE = 924747
        self.N_SIMULATED_TRACTS = 1000
        self.annotations_have_locations = ANNOTATIONS_HAVE_LOCATIONS
        self.use_simulated_data = SIMULATED_DATA
        self.VALID_ESTIMATE_PARAMETERS = ["p_y", "at_least_one_positive_image_by_area"]
        self.ESTIMATE_PARAMETERS = [ "p_y"]
        for p in self.ESTIMATE_PARAMETERS:
            assert p in self.VALID_ESTIMATE_PARAMETERS

        self.models = {
            "weighted_ICAR_prior": open("stan_models/weighted_ICAR_prior.stan").read(),
            "proper_car_prior": open("stan_models/proper_car_prior.stan").read(),
            "uniform_p_y": open(
                "stan_models/uniform_p_y_prior_just_for_debugging.stan"
            ).read(),
            "weighted_ICAR_prior_annotations_have_locations": open(
                "stan_models/weighted_ICAR_prior_annotations_have_locations.stan"
            ).read(),
        }

        self.logger = logger.setup_logger("ICAR_MODEL")
        self.logger.setLevel("INFO")
        self.logger.info("ICAR_MODEL instance initialized.")

        #self.adj_path = "data/processed/ct_nyc_adj_mtx.npy"
        self.adj_path = adj
        self.adj_matrix_storage = adj_matrix_storage

        self.RUNID = "NOT_SET"

    def inspect_data_to_use(self):
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
            self.data_to_use = util.read_real_data(
                annotations_have_locations=self.annotations_have_locations, adj=self.adj_path, adj_matrix_storage=self.adj_matrix_storage
            )
            self.logger.success("Successfully read empirical data.")

            # validate observed data
            observed_data_copy = self.inspect_data_to_use()
            util.validate_observed_data(
                observed_data_copy, self.annotations_have_locations
            )
            self.logger.success("Successfully validated the observed data.")
            del observed_data_copy



    def fit(self, CYCLES=1, WARMUP=1000, SAMPLES=1500):
        self.RUNID = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        os.makedirs(f"runs/{self.RUNID}", exist_ok=True)

        for i in range(CYCLES):
            self.load_data()

            if self.icar_prior_setting == "cheating":
                self.data_to_use["observed_data"]["use_ICAR_prior"] = 1
                self.data_to_use["observed_data"]["ICAR_prior_weight"] = 0.5
                if self.annotations_have_locations:
                    self.logger.info(
                        "Building model with weighted ICAR prior and annotations have locations."
                    )
                    model = stan.build(
                        self.models["weighted_ICAR_prior_annotations_have_locations"],
                        data=self.data_to_use["observed_data"],
                    )
                else:
                    self.logger.info("Building model with weighted ICAR prior.")
                    model = stan.build(
                        self.models["weighted_ICAR_prior"],
                        data=self.data_to_use["observed_data"],
                    )
            elif self.icar_prior_setting == "none":
                self.data_to_use["observed_data"]["use_ICAR_prior"] = 0
                self.data_to_use["observed_data"]["ICAR_prior_weight"] = 0
                if self.annotations_have_locations:
                    self.logger.info(
                        "Building model with no ICAR prior and annotations have locations."
                    )
                    model = stan.build(
                        self.models["weighted_ICAR_prior_annotations_have_locations"],
                        data=self.data_to_use["observed_data"],
                    )
                else:
                    self.logger.info("Building model with no ICAR prior.")
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

            fit = model.sample(num_chains=4, num_warmup=WARMUP, num_samples=SAMPLES)
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
                    ] + self.ESTIMATE_PARAMETERS,
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
                        ] + self.ESTIMATE_PARAMETERS,
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
                    ] + self.ESTIMATE_PARAMETERS,
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
                        ] + self.ESTIMATE_PARAMETERS,
                    ).to_string()
                )

        if self.use_simulated_data:

            for p in self.ESTIMATE_PARAMETERS:

                if p == "p_y":

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
                    plt.figure(figsize=[12, 3])

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
                    for k in param_names:
                        plt.subplot(1, len(param_names), param_names.index(k) + 1)
                        # histogram of posterior samples
                        plt.hist(df[k], bins=50, density=True)
                        plt.title(k)
                        plt.axvline(self.data_to_use["parameters"][k], color="red")
                        plt.savefig(f"runs/{self.RUNID}/simulated_params_histogram.png")
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
                        rf"Corr. between empirical $p(\\hat y = 1)$ and \n p\_y$, r = %.2f"
                        % pearsonr(empirical_estimate[~is_nan], estimate[~is_nan])[0]
                    )
                    plt.xlabel("empirical $p(\\hat y = 1)$")
                    plt.ylabel("inferred $p(y = 1)$")
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
            ax.hist(estimate, bins=200, density=True)
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

            # make df to write
            results = pd.DataFrame(
                {
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
    model = ICAR_MODEL(
        ICAR_PRIOR_SETTING="none",
        ANNOTATIONS_HAVE_LOCATIONS=False,
        SIMULATED_DATA=False,
        adj=["data/processed/ct_nyc_adj_list_node1.txt","data/processed/ct_nyc_adj_list_node2.txt"],
        adj_matrix_storage=False
    )
    #
    fit, df = model.fit(CYCLES=1, WARMUP=1000, SAMPLES=1500)
    model.plot_histogram(fit, df)
    model.plot_scatter(fit, df)
    model.plot_results(fit, df)
    model.write_estimate(fit, df)
    model.logger.success("All items in main program routine completed.")
