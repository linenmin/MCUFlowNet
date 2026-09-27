#!/bin/bash -l
set -euo pipefail
helper=$1
repo=$2
data=$3
runs=$4
previous_array=$5
models=(edge S L)
model=${models[${SLURM_ARRAY_TASK_ID:?}]}
if python3 "$helper" --runs "$runs" --model "$model" --predecessor "${previous_array}_${SLURM_ARRAY_TASK_ID}"; then
    exit 0
else
    result=$?
    [[ "$result" == 10 ]] || exit "$result"
fi
exec bash -l "$repo/tools/lowres/tier2.sh" train "$repo" "$data" "$runs"
