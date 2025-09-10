## Stan Models (overview)

### weighted_ICAR_prior.stan
- Data: edge lists `node1`, `node2`, counts per tract, and annotation totals.
- Prior: ICAR-like smoothing via `target += -ICAR_prior_weight * dot_self(phi[node1]-phi[node2])` when `use_ICAR_prior==1`.
- Parameters: `phi`, `phi_offset<=0`, ordered logits for detection rates conditioned on truth.
- Likelihood: Binomial for tract counts and annotation totals.

### ICAR_prior_annotations_have_locations.stan
- Extends ICAR with explicit annotation-location counts per tract and external covariates.
- Key parts:
  - External covariates matrix with coefficients `external_covariate_beta`.
  - Spatial component `phi_spatial_component` with proper ICAR penalty at weight 0.5.
  - Derived estimates: `at_least_one_positive_image_by_area` and `_if_you_have_100_images`.
- Likelihood splits annotated vs non-annotated contributions.

### proper_car_prior.stan
- Implements a proper CAR prior via `sparse_car_lpdf` using full adjacency matrix in transformed data.
- Parameters include `tau`, `alpha`, `phi_offset`, and detection probabilities.
- Requires `W` (NxN adjacency), `W_n`, and constructs sparse forms and eigenvalues.

### uniform_p_y_prior_just_for_debugging.stan
- No spatial structure; `p_y` is i.i.d. uniform on [0,1].
- Useful for sanity checks and debug runs.

Notes
- All models derive detection calibration terms from empirical `p_yhat`.
- The ICAR annotations-with-locations model is used when `ANNOTATIONS_HAVE_LOCATIONS=True` and supports external covariates.
- Model selection is controlled through `ICAR_PRIOR_SETTING` in `ICAR_MODEL`.
