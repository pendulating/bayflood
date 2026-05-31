#!/bin/bash

# Submit the full set of downsampling robustness runs for the post-adjacency-fix
# results. Two families, one run per ratio (matches the published methodology):
#
#   ANNOT (annotated-only downsampling): fracs 0.5 0.2 0.1 0.05 0.02  -> 2/5/10/20/50x
#   ALL   (all-images downsampling):     fracs 0.5 0.3333 0.25 0.2 0.1 -> 2/3/4/5/10x
#
# Each run is fit against the same full-data reference (the cov model in
# constants.CURRENT_DF). Run dirs land at
#   runs/icar_icar/simulated_False/ahl_True/covariates_True/<PREFIX>_<TIMESTAMP>/
# and the analysis CSV is analysis_df_<PREFIX>_<DATE>.csv. The notebook
# 02_downsampled_all_performance.ipynb discovers these by PREFIX glob.
#
# Each run uses a distinct, fixed seed so the random binomial downsample is
# reproducible AND independent across ratios (re-running any job reproduces it
# exactly; the recorded seed is written to the run's metadata.json).
#
# Jobs are chained (each waits for the previous via afterany) so that only one
# 400 GB job runs at a time. To run them in parallel instead, delete the
# --dependency line below.
#
# Usage:
#   export REPO_ROOT=/share/ju/matt/bayflood
#   bash jobs/downsampled_sequence.sh

set -euo pipefail

: "${REPO_ROOT:=/share/ju/matt/bayflood}"
export REPO_ROOT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GEOMETRY=ct
TAG=DOWNSAMPLE_JUN03

mkdir -p "$REPO_ROOT/.slurm_jobs/fit"

# (sub-script, prefix, frac, seed) -- distinct seed per run
JOBS=(
    "downsampled.sub      ${TAG}_ANNOT_0.5   0.5     101"
    "downsampled.sub      ${TAG}_ANNOT_0.2   0.2     102"
    "downsampled.sub      ${TAG}_ANNOT_0.1   0.1     103"
    "downsampled.sub      ${TAG}_ANNOT_0.05  0.05    104"
    "downsampled.sub      ${TAG}_ANNOT_0.02  0.02    105"
    "downsampled_all.sub  ${TAG}_ALL_0.5     0.5     201"
    "downsampled_all.sub  ${TAG}_ALL_0.3333  0.3333  202"
    "downsampled_all.sub  ${TAG}_ALL_0.25    0.25    203"
    "downsampled_all.sub  ${TAG}_ALL_0.2     0.2     204"
    "downsampled_all.sub  ${TAG}_ALL_0.1     0.1     205"
)

previous_job_id=""
for entry in "${JOBS[@]}"; do
    read -r sub prefix frac seed <<< "$entry"
    if [ -z "$previous_job_id" ]; then
        job_id=$(sbatch "$SCRIPT_DIR/$sub" "$prefix" "$GEOMETRY" "$frac" "$seed" | awk '{print $NF}')
    else
        job_id=$(sbatch --dependency=afterany:"$previous_job_id" "$SCRIPT_DIR/$sub" "$prefix" "$GEOMETRY" "$frac" "$seed" | awk '{print $NF}')
    fi
    echo "Submitted $sub  prefix=$prefix  frac=$frac  seed=$seed  ->  job $job_id"
    previous_job_id=$job_id
done
