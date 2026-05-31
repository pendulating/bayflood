#!/bin/bash

# Resubmit the all-images downsampling runs that aborted on non-convergence
# (r_hat > 1.1) in the DOWNSAMPLE_JUN03 batch:
#
#   ALL_0.5  (2x)  seed 201 -> r_hat 1.20 on p_y[1085]
#   ALL_0.3333 (3x) seed 202 -> r_hat 1.31 on p_y[1085]
#   ALL_0.1  (10x) seed 205 -> r_hat 1.44 on spatial_sigma
#
# Heavily-downsampled all-image fits carry less information and mix less
# reliably; a fresh seed draws a different thinned dataset. New prefixes/seeds
# so these do not collide with the (kept) converged runs. The notebook
# 02_downsampled_all_performance.ipynb discovers by prefix DOWNSAMPLE_JUN03_ALL_*
# and takes the most recent converged run per frac, so once these converge they
# supersede nothing and simply fill the 2x/3x/10x gaps.
#
# Usage:
#   bash jobs/downsampled_all_resubmit.sh

set -euo pipefail

: "${REPO_ROOT:=/share/ju/matt/bayflood}"
export REPO_ROOT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GEOMETRY=ct
TAG=DOWNSAMPLE_JUN03

mkdir -p "$REPO_ROOT/.slurm_jobs/fit"

# (prefix, frac, seed) -- new seeds so the redraw differs from the failed run
JOBS=(
    "${TAG}_ALL_0.5     0.5     211"
    "${TAG}_ALL_0.3333  0.3333  212"
    "${TAG}_ALL_0.1     0.1     215"
)

previous_job_id=""
for entry in "${JOBS[@]}"; do
    read -r prefix frac seed <<< "$entry"
    if [ -z "$previous_job_id" ]; then
        job_id=$(sbatch "$SCRIPT_DIR/downsampled_all.sub" "$prefix" "$GEOMETRY" "$frac" "$seed" | awk '{print $NF}')
    else
        job_id=$(sbatch --dependency=afterany:"$previous_job_id" "$SCRIPT_DIR/downsampled_all.sub" "$prefix" "$GEOMETRY" "$frac" "$seed" | awk '{print $NF}')
    fi
    echo "Submitted downsampled_all.sub  prefix=$prefix  frac=$frac  seed=$seed  ->  job $job_id"
    previous_job_id=$job_id
done
