#!/bin/bash -l
# Arguments: mode repo data experiment-root; arrays 0..2 probe, 0..5 train.
set -euo pipefail
mode=$1 repo=$2 data=$3 root=$4
cd "$repo"
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
source "$HOME/tf_work/bin/activate"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$VIRTUAL_ENV/lib/python3.11/site-packages"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export TF_DETERMINISTIC_OPS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
models=(edge S L)
index=${SLURM_ARRAY_TASK_ID:?}
model=${models[$((index % 3))]}
if [[ "$mode" == probe ]]; then
    python tools/lowres/verify_comparison.py --model "$model" --data "$data" \
        --manifests "$root/manifests" --init "$root/source/$model/fc2" --out "$root/probe/$model"
elif [[ "$mode" == train ]]; then
    python -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"]' "$root/probe/$model/acceptance.json"
    schedule=constant
    (( index < 3 )) || schedule=cosine
    out="$root/$schedule/seed42/$model/ft3d"
    if [[ -n "${5:-}" ]]; then
        set +e
        python tools/lowres/continue.py --runs "$root" --model "$model" \
            --phase-out "$out" --epochs 20 --predecessor "${5}_${index}"
        decision=$?
        set -e
        [[ "$decision" != 0 ]] || exit 0
        [[ "$decision" == 10 ]] || exit "$decision"
    fi
    args=(--model "$model" --phase ft3d --data "$data" --manifests "$root/manifests"
          --init "$root/source/$model/fc2" --out "$out" --epochs 20 --seed 42
          --merge-tail --keep-every 5 --lr-schedule "$schedule" --min-lr 1e-6)
    [[ ! -f "$out/current.json" ]] || args+=(--resume)
    python tools/lowres/train.py "${args[@]}"
else
    echo "Unknown mode: $mode" >&2; exit 2
fi
