#!/bin/bash -l
# Three inherited models; finite FC2 adaptation only, with an on-server gate.
set -euo pipefail
mode=$1 repo=$2 data=$3 root=$4
[[ "${SLURM_CLUSTER_NAME:-}" == mindwell ]]
software="$VSC_SCRATCH/MCUFlowNet/software"
[[ -f "$software/READY" && -f "$root/control/SOURCES_READY.json" ]]
cd "$repo"
commit=$(git rev-parse HEAD)
run_python() {
    apptainer exec --nv --cleanenv \
        --bind "$software:$software" --bind "$repo:$repo:ro" \
        --bind "$data:$data:ro" --bind "$root:$root" --pwd "$repo" \
        --env TF_USE_LEGACY_KERAS=1 --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 \
        --env PYTHONUNBUFFERED=1 --env TF_DETERMINISTIC_OPS=1 --env TF_CPP_MIN_LOG_LEVEL=2 \
        --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env OMP_NUM_THREADS=8 --env OPENBLAS_NUM_THREADS=1 \
        --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?}" \
        "$software/containers/tensorflow-25.02.sif" "$software/tf2502/bin/python" "$@"
}
models=(edge S L)
index=${SLURM_ARRAY_TASK_ID:?}
[[ "$index" =~ ^[0-2]$ ]]
model=${models[$index]}
source="$root/source/$model/model"
probe="$root/probe/$model"
if [[ "$mode" == probe ]]; then
    run_python tools/lowres/test_data.py
    run_python tools/lowres/test_protocol.py
    run_python tools/lowres/verify_adaptation.py --model "$model" --checkpoint "$source" \
        --data "$data" --manifests "$root/manifests" --out "$probe" --code-commit "$commit"
elif [[ "$mode" == train ]]; then
    run_python -c 'import json,sys; v=json.load(open(sys.argv[1])); assert v["passed"] and v["source_unchanged"]' "$probe/acceptance.json"
    if [[ -n "${5:-}" ]]; then
        set +e
        python3 tools/lowres/continue.py --runs "$root" --model "$model" \
            --epochs 20 --final-phase fc2 --predecessor "${5}_${index}"
        decision=$?
        set -e
        [[ "$decision" != 0 ]] || exit 0
        [[ "$decision" == 10 ]] || exit "$decision"
    fi
    out="$root/seed42/$model/fc2"
    args=(--model "$model" --phase fc2 --data "$data" --manifests "$root/manifests" \
          --out "$out" --init-checkpoint "$source" --epochs 20 --seed 42 \
          --initial-lr 1e-5 --lr-schedule cosine --min-lr 1e-6 --eval-every 1 --keep-every 5)
    [[ "$model" != edge ]] || args+=(--edge-public --bn-mode frozen)
    [[ ! -f "$out/current.json" ]] || args+=(--resume)
    run_python tools/lowres/train.py "${args[@]}" --code-commit "$commit"
else
    echo "Unknown mode: $mode" >&2; exit 2
fi
