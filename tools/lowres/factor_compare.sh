#!/bin/bash -l
# Four controls: large Edge/S/L, then small Edge with frozen BN statistics.
set -euo pipefail
mode=$1 repo=$2 data=$3 root=$4
cd "$repo"
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
source "$HOME/tf_work/bin/activate"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$VIRTUAL_ENV/lib/python3.11/site-packages"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export TF_DETERMINISTIC_OPS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
index=${SLURM_ARRAY_TASK_ID:?}
models=(edge S L edge)
model=${models[$index]}
variant=large height=320 width=416 bn=train
if [[ "$index" == 3 ]]; then variant=frozen; height=160; width=208; bn=frozen; fi
probe="$root/probe/$variant-$model"
if [[ "$mode" == probe ]]; then
    python tools/lowres/test_data.py
    python tools/lowres/test_protocol.py
    python tools/lowres/verify_factors.py --model "$model" --height "$height" --width "$width" \
        --bn-mode "$bn" --data "$data" --manifests "$root/manifests" --out "$probe" --fc2-only
elif [[ "$mode" == train ]]; then
    python -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"]' "$probe/acceptance.json"
    if [[ -n "${5:-}" ]]; then
        set +e
        python tools/lowres/continue.py --runs "$root/$variant" --model "$model" \
            --epochs 50 --final-phase fc2 --predecessor "${5}_${index}"
        decision=$?
        set -e
        [[ "$decision" != 0 ]] || exit 0
        [[ "$decision" == 10 ]] || exit "$decision"
    fi
    for phase in fc2; do
        out="$root/$variant/seed42/$model/$phase"
        args=(--model "$model" --phase "$phase" --data "$data" --manifests "$root/manifests"
              --out "$out" --seed 42 --height "$height" --width "$width" --bn-mode "$bn")
        args+=(--epochs 50 --lr-schedule constant --keep-every 10)
        [[ ! -f "$out/current.json" ]] || args+=(--resume)
        python tools/lowres/train.py "${args[@]}"
    done
else
    echo "Unknown mode: $mode" >&2; exit 2
fi
