#!/bin/bash -l
# Common scratch-initialized FC2 experiment; six paired geometry arms.
set -euo pipefail
mode=$1 repo=$2 data=$3 root=$4
[[ "${SLURM_CLUSTER_NAME:-}" == mindwell ]]
software="$VSC_SCRATCH/MCUFlowNet/software"
[[ -f "$software/READY" && -f "$root/control/SOURCES_READY.json" ]]
cd "$repo"
commit=$(git rev-parse HEAD)
[[ "$commit" == "$(cat "$root/control/code-commit.txt")" ]]
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
index=${SLURM_ARRAY_TASK_ID:?}
if [[ "$mode" == probe ]]; then
    models=(edge S L)
    [[ "$index" =~ ^[0-2]$ ]]
    model=${models[$index]}
    run_python tools/lowres/test_data.py
    run_python tools/lowres/test_geometry.py
    run_python tools/lowres/verify_geometry.py --model "$model" \
        --data "$data" --manifests "$root/manifests" --source "$root/source/$model/fc2" \
        --out "$root/probe/$model" --code-commit "$commit"
elif [[ "$mode" == train ]]; then
    models=(edge S L edge S L); arms=(whole whole whole random random random)
    [[ "$index" =~ ^[0-5]$ ]]
    model=${models[$index]} arm=${arms[$index]}
    run_python -c 'import json,sys; a=json.load(open(sys.argv[1])); assert a["passed"] and a["source_unchanged"] and a["code_commit"]==sys.argv[2]' \
        "$root/probe/$model/acceptance.json" "$commit"
    out="$root/seed42/$arm/$model/fc2"
    if [[ -n "${5:-}" ]]; then
        if [[ -f "$out/status.json" ]] && python3 -c 'import json,sys; a=json.load(open(sys.argv[1])); assert a["completed"] and a["step"]==10000' "$out/status.json"; then
            printf 'All 10000 steps complete; no training needed.\n'; exit 0
        fi
        previous="${5}_${index}"
        predecessor_state=$(sacct --clusters=mindwell -X -nP -j "$previous" -o JobIDRaw,State | awk -F'|' -v id="$previous" '$1==id {print $2}')
        case "$predecessor_state" in
            TIMEOUT*|NODE_FAIL*|PREEMPTED*) ;;
            *) printf 'No retry for %s state=%s\n' "$previous" "$predecessor_state"; exit 2 ;;
        esac
    fi
    args=(--model "$model" --geometry "$arm" --data "$data" --manifests "$root/manifests" \
          --source "$root/source/$model/fc2" --out "$out" --steps 10000 --eval-every 1000 --seed 42)
    [[ ! -f "$out/current.json" ]] || args+=(--resume)
    run_python tools/lowres/geometry_compare.py "${args[@]}" --code-commit "$commit"
else
    printf 'Unknown mode: %s\n' "$mode" >&2; exit 2
fi
