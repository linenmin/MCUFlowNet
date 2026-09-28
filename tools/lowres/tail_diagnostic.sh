#!/bin/bash -l
set -euo pipefail
repo=$1
data=$2
root=$3
cd "$repo"
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
source "$HOME/tf_work/bin/activate"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$VIRTUAL_ENV/lib/python3.11/site-packages"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export TF_DETERMINISTIC_OPS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
models=(edge S L)
model=${models[${SLURM_ARRAY_TASK_ID:?}]}
args=(--model "$model" --data "$data" --manifests "$root/manifests"
      --source "$root/source/$model/model" --source-state "$root/source/$model/current.json")
# Gate each full diagnostic on its own tiny real-data restore/branch test.
python tools/lowres/tail_diagnostic.py "${args[@]}" --probe --out "$root/probe/$model"
python tools/lowres/tail_diagnostic.py "${args[@]}" --out "$root/results/$model"
