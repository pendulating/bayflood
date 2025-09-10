## Stan Models (overview)

### ICAR_prior_annotations_have_locations.stan
- Extends ICAR with explicit annotation-location counts per tract and external covariates.
- Key parts:
  - External covariates matrix with coefficients `external_covariate_beta`.
  - Spatial component `phi_spatial_component` with proper ICAR penalty at weight 0.5.
  - Derived estimates: `at_least_one_positive_image_by_area` and `_if_you_have_100_images`.
- Likelihood splits annotated vs non-annotated contributions.
Notes
- All models derive detection calibration terms from empirical `p_yhat`.
- The ICAR annotations-with-locations model is used when `ANNOTATIONS_HAVE_LOCATIONS=True` and supports external covariates.
 - This artifact uses the ICAR family only: `ICAR_prior_annotations_have_locations.stan` when annotation locations exist, and `weighted_ICAR_prior.stan` otherwise.
