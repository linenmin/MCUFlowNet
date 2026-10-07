#!/bin/bash -l
set -euo pipefail
mode=$1 repo=$2 data=$3 root=$4
[[ "${SLURM_CLUSTER_NAME:?}" == mindwell || "$SLURM_CLUSTER_NAME" == wice ]]
project="${root%/runs/LOWRES-BENCH-01/*}"
[[ "$project" != "$root" && "$(realpath "$project")" == "$project" ]]
software="$project/software"
[[ -f "$software/READY" && -f "$root/control/READY.json" ]]
cd "$repo";commit=$(git rev-parse HEAD)
[[ "$commit" == "$(cat "$root/control/code-commit.txt")" && -z "$(git status --porcelain)" ]]
run_python() {
    apptainer exec --nv --cleanenv --bind "$software:$software" --bind "$repo:$repo:ro" --bind "$data:$data:ro" \
      --bind "$root:$root" --bind "$root/source:$root/source:ro" --pwd "$repo" \
      --env TF_USE_LEGACY_KERAS=1 --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 \
      --env TF_DETERMINISTIC_OPS=1 --env TF_CPP_MIN_LOG_LEVEL=2 --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      --env OMP_NUM_THREADS=8 --env OPENBLAS_NUM_THREADS=1 --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?}" \
      "$software/containers/tensorflow-25.02.sif" "$software/tf2502/bin/python" "$@"
}
index=${SLURM_ARRAY_TASK_ID:?};[[ "$index" =~ ^[01]$ ]];models=(S L);model=${models[$index]}
if [[ "$mode" == probe ]];then
    run_python tools/lowres/verify_geometry.py --final-sl --model "$model" --data "$data" \
      --manifests "$root/manifests" --source "$root/source/$model/fc2" --out "$root/probe/$model" --code-commit "$commit"
elif [[ "$mode" == train ]];then
    python3 - "$root/probe/$model/acceptance.json" "$commit" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]));assert a['passed'] and a['source_unchanged'] and a['code_commit']==sys.argv[2]
assert a['planned_steps']==40000 and a['full_sintel_best_selection'] and a['continuous_resumed_all_variables_exact']
assert a['actual_gpu_backprop'] and a['adam_reset_at_stage_start']
PY
    out="$root/seed42/mixture75_25/$model/replay"
    if [[ -f "$out/status.json" ]] && python3 - "$out/status.json" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]));sys.exit(0 if a['completed'] and a['step']==40000 and a['total_steps']==40000 else 1)
PY
    then printf 'Fixed40000 budget already completed.\n';exit 0;fi
    if [[ -n "${5:-}" ]];then
        [[ "$5" =~ ^[0-9]+$ ]];previous="${5}_${index}"
        state=$(sacct --clusters="$SLURM_CLUSTER_NAME" -X -nP -j "$previous" -o JobID,State | awk -F'|' -v id="$previous" '$1==id {print $2}')
        case "$state" in TIMEOUT*|NODE_FAIL*|PREEMPTED*) ;; *) printf 'No retry: %s %s\n' "$previous" "$state";exit 2;; esac
    fi
    args=(--model "$model" --geometry random --phase fc2 --fc2-source-step 10000 --replay-arm mixture75_25 \
      --select-full-sintel --data "$data" --manifests "$root/manifests" --source "$root/source/$model/fc2" \
      --out "$out" --steps 40000 --eval-every 1000 --initial-lr 3e-6 --min-lr 1e-6 --seed 42 --workers 8 --code-commit "$commit")
    [[ ! -f "$out/current.json" ]] || args+=(--resume)
    run_python tools/lowres/geometry_compare.py "${args[@]}"
else printf 'Unknown mode: %s\n' "$mode" >&2;exit 2;fi
