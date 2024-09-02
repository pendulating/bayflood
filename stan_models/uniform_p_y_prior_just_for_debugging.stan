data {
  int<lower=0> N; // number of Census tracts. 
  array[N] int<lower=0> n_images_by_area; // vector with one entry per Census tract of the number of images in that tract. 
  array[N] int<lower=0> n_classified_positive_by_area; // vector with one entry per Census tract of number of images classified positive. 

  //annotation sample. 
  int total_annotated_classified_negative;
  int total_annotated_classified_positive;
  int total_annotated_classified_negative_true_positive;
  int total_annotated_classified_positive_true_positive;  
}
parameters {
  vector<lower=0, upper=1>[N] p_y; // probability of a positive image in each Census tract.
  ordered[2] logit_p_y_1_given_y_hat; // ordered to impose the constraint that p_y_1_given_y_hat_0 < p_y_1_given_y_hat_1.
}
transformed parameters {
    real p_y_1_given_y_hat_0 = inv_logit(logit_p_y_1_given_y_hat[1]);
    real p_y_1_given_y_hat_1 = inv_logit(logit_p_y_1_given_y_hat[2]);
    real empirical_p_yhat = sum(n_classified_positive_by_area) * 1.0 / sum(n_images_by_area);
    real p_y_hat_1_given_y_1 = empirical_p_yhat * p_y_1_given_y_hat_1 / (empirical_p_yhat * p_y_1_given_y_hat_1 + (1 - empirical_p_yhat) * p_y_1_given_y_hat_0);
    real p_y_hat_1_given_y_0 = empirical_p_yhat * (1 - p_y_1_given_y_hat_1) / (empirical_p_yhat * (1 - p_y_1_given_y_hat_1) + (1 - empirical_p_yhat) * (1 - p_y_1_given_y_hat_0));
}
model {
  // model the results on the annotation set. 
  total_annotated_classified_negative_true_positive ~ binomial(total_annotated_classified_negative, p_y_1_given_y_hat_0);
  total_annotated_classified_positive_true_positive ~ binomial(total_annotated_classified_positive, p_y_1_given_y_hat_1);
  
  // model the results by Census tract. 
  p_y ~ uniform(0, 1);
  n_classified_positive_by_area ~ binomial(n_images_by_area, p_y .* p_y_hat_1_given_y_1 + (1 - p_y) .* p_y_hat_1_given_y_0);
}