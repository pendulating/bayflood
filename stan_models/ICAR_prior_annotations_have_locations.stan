
// Stan code below works and has been reviewed but does not implement a proper ICAR prior. Set variable use_ICAR_prior to 0 if you don't want to use one at all. 
// I think it might also be principled to set the ICAR prior weight to 0.5, https://mc-stan.org/users/documentation/case-studies/icar_stan.html. 
// This assumes we have location information for each annotation. 
data {
  int<lower=0> N; // number of Census tracts. 
  int<lower=0> N_edges; // number of edges in the graph (i.e. number of pairs of adjacent Census tracts). 
  array[N_edges] int<lower=1, upper=N> node1; // node1[i] adjacent to node2[i]
  array[N_edges] int<lower=1, upper=N> node2; // and node1[i] < node2[i]
  vector[N] n_images_by_area; // vector with one entry per Census tract of the number of images in that tract. 
  vector[N] n_classified_positive_by_area; // vector with one entry per Census tract of number of images classified positive. 
  vector[N] n_classified_positive_annotated_positive_by_area;
  vector[N] n_classified_positive_annotated_negative_by_area;
  vector[N] n_classified_negative_annotated_positive_by_area;
  vector[N] n_classified_negative_annotated_negative_by_area;
  array[N] int<lower=0> n_non_annotated_by_area;
  array[N] int<lower=0> n_non_annotated_by_area_classified_positive;
  real <upper=0> center_of_phi_offset_prior; 

  int<lower=0,upper=1> use_ICAR_prior; // 1 if you want to use ICAR prior, 0 if you don't. ICAR prior basically smooths the data. 

  // external covariate matrix. This part is new. 
  int<lower=1> n_external_covariates; // number of external covariates.
  matrix[N, n_external_covariates] external_covariates; // matrix with one row per Census tract and one column per external covariate.
}
parameters {
  //real<upper=0> phi_offset; // this is the mean from which phis are drawn. Upper bound at 0 to rule out bad modes and set prior that true positives are rare. 
  ordered[2] logit_p_y_1_given_y_hat; // ordered to impose the constraint that p_y_1_given_y_hat_0 < p_y_1_given_y_hat_1.
  real<lower=0> spatial_sigma; 
  vector[N] phi_spatial_component;
  vector[n_external_covariates] external_covariate_beta; // coefficients for the external covariates.
}
transformed parameters {
    vector[N] phi = center_of_phi_offset_prior + external_covariates * external_covariate_beta + phi_spatial_component * spatial_sigma; 
    real p_y_1_given_y_hat_0 = inv_logit(logit_p_y_1_given_y_hat[1]);
    real p_y_1_given_y_hat_1 = inv_logit(logit_p_y_1_given_y_hat[2]);
    vector[N] p_y = inv_logit(phi);
    real empirical_p_yhat = sum(n_classified_positive_by_area) * 1.0 / sum(n_images_by_area);
    real p_y_hat_1_given_y_1 = empirical_p_yhat * p_y_1_given_y_hat_1 / (empirical_p_yhat * p_y_1_given_y_hat_1 + (1 - empirical_p_yhat) * p_y_1_given_y_hat_0);
    real p_y_hat_1_given_y_0 = empirical_p_yhat * (1 - p_y_1_given_y_hat_1) / (empirical_p_yhat * (1 - p_y_1_given_y_hat_1) + (1 - empirical_p_yhat) * (1 - p_y_1_given_y_hat_0));
    vector[N] at_least_one_positive_image_by_area = (1 - pow(1 - p_y, n_images_by_area));
    vector[N] at_least_one_positive_image_by_area_if_you_have_100_images = (1 - pow(1 - p_y, 100));
    // set at_least_one_positive_image to 1 if there is at least one annotated positive image in the Census tract.
    for(i in 1:N) {
      if ((n_classified_positive_annotated_positive_by_area[i] > 0) || (n_classified_negative_annotated_positive_by_area[i] > 0)) {
        at_least_one_positive_image_by_area[i] = 1;
        at_least_one_positive_image_by_area_if_you_have_100_images[i] = 1;
      }
    }
}
model {
  // You can't just scale ICAR priors by random numbers; the only principled value for ICAR_prior_weight is 0.5. 
  // https://stats.stackexchange.com/questions/333258/strength-parameter-in-icar-spatial-model
  // see https://mc-stan.org/users/documentation/case-studies/icar_stan.html for source. 
  if (use_ICAR_prior == 1) {
    // just have the spatial component with an L2 loss tying adjacent areas together. 
    target += -0.5 * dot_self(phi_spatial_component[node1] - phi_spatial_component[node2]);
    sum(phi_spatial_component) ~ normal(0, 0.001 * N);
    spatial_sigma ~ normal(0, 1);
  }else{
    // now the spatial effects are just random effects (adjacent areas are uncorrelated, no smoothing). 
    phi_spatial_component ~ normal(0, 1); 
    spatial_sigma ~ normal(0, 1);
  }
  
  external_covariate_beta ~ normal(0, 2); // we no longer need an explicit phi offset because it's wrapped into the intercept term. 
  
  // model the classification results by Census tract for the UNANNOTATED images. 
  // note that this binomial statement should be equivalent a statement which increments the target directly, but that's more verbose. 
  n_non_annotated_by_area_classified_positive ~ binomial(n_non_annotated_by_area, p_y .* p_y_hat_1_given_y_1 + (1 - p_y) .* p_y_hat_1_given_y_0);
  
  // model the annotation results by Census tract. 
  target += (sum(n_classified_positive_annotated_positive_by_area .* log(p_y * p_y_hat_1_given_y_1))
      + sum(n_classified_positive_annotated_negative_by_area .* log((1 - p_y) * p_y_hat_1_given_y_0))
    + sum(n_classified_negative_annotated_positive_by_area .* log(p_y * (1 - p_y_hat_1_given_y_1)))
    + sum(n_classified_negative_annotated_negative_by_area .* log((1 - p_y) * (1 - p_y_hat_1_given_y_0))));  
}
