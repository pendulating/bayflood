#!/bin/bash

# Submit multiple baseline comparison jobs in sequence
# Each job waits for the previous to complete successfully

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 NUMBER_OF_RUNS"
    exit 1
fi

N=$1
previous_job_id=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create log directory if it doesn't exist
mkdir -p /share/ju/matt/bayflood/.slurm_jobs/baselines

for i in $(seq 1 $N); do
    if [ -z "$previous_job_id" ]; then
        # Submit first job without dependency
        job_id=$(sbatch "$SCRIPT_DIR/baselines.sub" | cut -d ' ' -f 4)
    else
        # Submit subsequent jobs with dependency on previous job
        job_id=$(sbatch --dependency=afterok:$previous_job_id "$SCRIPT_DIR/baselines.sub" | cut -d ' ' -f 4)
    fi
    
    echo "Submitted job $i with ID: $job_id"
    previous_job_id=$job_id
done