#!/bin/bash -l
# Resource/account/log arguments are explicit at submission and recorded there.
set -euo pipefail
mode=$1
repo=$2
data=$3
runs=$4
cd "$repo"
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
source "$HOME/tf_work/bin/activate"
export PYTHONPATH="$VIRTUAL_ENV/lib/python3.11/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TF_DETERMINISTIC_OPS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
if [[ "$mode" == prepare ]]; then
    python tools/lowres/test_data.py
    python tools/lowres/data.py --root "$data" --out "$runs/manifests"
    exit
fi
nvidia-smi
models=(edge S L)
model=${models[${SLURM_ARRAY_TASK_ID:?}]}
if [[ "$mode" == probe ]]; then
    python tools/lowres/verify.py --model "$model" --data "$data" --manifests "$runs/manifests" --out "$runs/probe/$model"
elif [[ "$mode" == train ]]; then
    test -f "$runs/probe/$model/acceptance.json"
    for phase in fc2 ft3d; do
        out="$runs/seed42/$model/$phase"
        args=(--model "$model" --phase "$phase" --data "$data" --manifests "$runs/manifests" --out "$out")
        [[ "$phase" == ft3d ]] && args+=(--init "$runs/seed42/$model/fc2")
        [[ -f "$out/current.json" ]] && args+=(--resume)
        python tools/lowres/train.py "${args[@]}"
    done
else
    echo "Unknown mode: $mode" >&2; exit 2
fi
