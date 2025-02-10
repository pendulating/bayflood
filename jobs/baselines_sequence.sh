#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 NUMBER_OF_RUNS"
    exit 1
fi

N=$1
previous_job_id=""

for i in $(seq 1 $N); do
    if [ -z "$previous_job_id" ]; then
        # Submit first job without dependency
        job_id=$(sbatch baselines.sub | cut -d ' ' -f 4)
    else
        # Submit subsequent jobs with dependency on previous job
        job_id=$(sbatch --dependency=afterok:$previous_job_id baselines.sub | cut -d ' ' -f 4)
    fi
    
    echo "Submitted job $i with ID: $job_id"
    previous_job_id=$job_id
done