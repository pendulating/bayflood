## CLI Reference

### icar_model.py

```
python icar_model.py <icar_prior_setting> \
  [--annotations_have_locations] \
  [--simulated_data] \
  [--external_covariates] \
  [--prefix STR] \
  [--downsample_frac FLOAT] \
  [--empirical_data_path PATH] \
  [--adj_node1_path PATH] [--adj_node2_path PATH] \
  [--adj_npy_path PATH] \
  [--compare_to_baselines]
```

- `icar_prior_setting`: `none | icar | proper | just_model_p_y`
- `--empirical_data_path`: overrides `config.DATASET_PATH` / `EMPIRICAL_DATA_PATH`
- `--adj_*`: specify adjacency inputs (edge lists or `.npy`)

### generate_maps.py (optional)

```
python generate_maps.py <run_id> <estimate_path> <estimate>
```

- `estimate`: `p_y` or `at_least_one_positive_image_by_area`
