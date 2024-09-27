functions {
  /**
  * Stan code with CAR prior seems to work but hasn't been reviewed. 
  * CAR prior is https://mc-stan.org/users/documentation/case-studies/mbjoseph-CARStan.html
  * Return the log probability of a proper conditional autoregressive (CAR) prior 
  * with a sparse representation for the adjacency matrix
  *
  * @param phi Vector containing the parameters with a CAR prior
  * @param tau Precision parameter for the CAR prior (real)
  * @param alpha Dependence (usually spatial) parameter for the CAR prior (real)
  * @param W_sparse Sparse representation of adjacency matrix (int array)
  * @param n Length of phi (int)
  * @param W_n Number of adjacent pairs (int)
  * @param D_sparse Number of neighbors for each location (vector)
  * @param lambda Eigenvalues of D^{-1/2}*W*D^{-1/2} (vector)
  *
  * @return Log probability density of CAR prior up to additive constant
  */
  real sparse_car_lpdf(vector phi, real tau, real alpha,
                       array[,] int W_sparse, vector D_sparse, vector lambda,
                       int n, int W_n) {
    row_vector[n] phit_D; // phi' * D
    row_vector[n] phit_W; // phi' * W
    vector[n] ldet_terms;
    
    phit_D = (phi .* D_sparse)';
    phit_W = rep_row_vector(0, n);
    for (i in 1 : W_n) {
      phit_W[W_sparse[i, 1]] = phit_W[W_sparse[i, 1]] + phi[W_sparse[i, 2]];
      phit_W[W_sparse[i, 2]] = phit_W[W_sparse[i, 2]] + phi[W_sparse[i, 1]];
    }
    
    for (i in 1 : n) {
      ldet_terms[i] = log1m(alpha * lambda[i]);
    }
    return 0.5
           * (n * log(tau) + sum(ldet_terms)
              - tau * (phit_D * phi - alpha * (phit_W * phi)));
  }
}

data {
  int<lower=0> N;
  int<lower=0> N_edges;
  matrix<lower=0, upper=1>[N, N] W; // adjacency matrix
  int W_n; // number of adjacent region pairs
  array[N] n_images_by_area; 
  array[N] n_classified_positive_by_area; 

  //annotation sample. 
  int total_annotated_classified_negative;
  int total_annotated_classified_positive;
  int total_annotated_classified_negative_true_positive;
  int total_annotated_classified_positive_true_positive;  
}
transformed data {
  array[W_n, 2] int W_sparse; // adjacency pairs
  vector[N] D_sparse; // diagonal of D (number of neigbors for each site)
  vector[N] lambda; // eigenvalues of invsqrtD * W * invsqrtD
  
  {
    // generate sparse representation for W
    int counter;
    counter = 1;
    // loop over upper triangular part of W to identify neighbor pairs
    for (i in 1 : (N - 1)) {
      for (j in (i + 1) : N) {
        if (W[i, j] == 1) {
          W_sparse[counter, 1] = i;
          W_sparse[counter, 2] = j;
          counter = counter + 1;
        }
      }
    }
  }
  for (i in 1 : N) {
    D_sparse[i] = sum(W[i]);
  }
  {
    vector[N] invsqrtD;
    for (i in 1 : N) {
      invsqrtD[i] = 1 / sqrt(D_sparse[i]);
    }
    lambda = eigenvalues_sym(quad_form(W, diag_matrix(invsqrtD)));
  }
}

parameters {
  vector[N] phi;
  real<lower=0> tau;
  real<lower=0, upper=1> alpha;
  real <upper=0>phi_offset; // you may not want to place this constraint but it helps convergence with small numbers of samples
  real<lower=0, upper=1> p_y_1_given_y_hat_1; 
  real<lower=0, upper=1> p_y_1_given_y_hat_0;
}
transformed parameters {
    vector[N] p_y = inv_logit(phi + phi_offset);
    //real predicted_overall_p_y = sum(n_images_by_area .* p_y) / sum(n_images_by_area);
    real empirical_p_yhat = sum(n_classified_positive_by_area) * 1.0 / sum(n_images_by_area);
    real p_y_hat_1_given_y_1 = empirical_p_yhat * p_y_1_given_y_hat_1 / (empirical_p_yhat * p_y_1_given_y_hat_1 + (1 - empirical_p_yhat) * p_y_1_given_y_hat_0);
    real p_y_hat_1_given_y_0 = empirical_p_yhat * (1 - p_y_1_given_y_hat_1) / (empirical_p_yhat * (1 - p_y_1_given_y_hat_1) + (1 - empirical_p_yhat) * (1 - p_y_1_given_y_hat_0));
}
model {
  total_annotated_classified_negative_true_positive ~ binomial(total_annotated_classified_negative , p_y_1_given_y_hat_0);
  total_annotated_classified_positive_true_positive ~ binomial(total_annotated_classified_positive, p_y_1_given_y_hat_1);
  tau ~ gamma(2, 2);
  phi ~ sparse_car(tau, alpha, W_sparse, D_sparse, lambda, N, W_n);
  phi_offset ~ normal(-4, 0.5);
  n_classified_positive_by_area ~ binomial(n_images_by_area, p_y .* p_y_hat_1_given_y_1 + (1 - p_y) .* p_y_hat_1_given_y_0);
}