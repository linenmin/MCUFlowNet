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
read -r steps initial_lr models_csv < <(python3 - "$root/control/submission.json" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]));assert a['approved'] and a['source_step']==10000
steps=a['steps'];initial=a.get('initial_lr',3e-6)
assert (a['recipe_id'],steps,initial,tuple(a['models'])) in [('FINAL-SL-02',40000,3e-6,('S','L')),('FINAL-SL-04',80000,3e-5,('S','L')),('FINAL-EDGE-04',80000,3e-5,('edge',))]
assert a.get('min_lr',1e-6)==1e-6
print(steps,initial,','.join(a['models']))
PY
)
run_python() {
    apptainer exec --nv --cleanenv --bind "$software:$software" --bind "$repo:$repo:ro" --bind "$data:$data:ro" \
      --bind "$root:$root" --bind "$root/source:$root/source:ro" --pwd "$repo" \
      --env TF_USE_LEGACY_KERAS=1 --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 \
      --env TF_DETERMINISTIC_OPS=1 --env TF_CPP_MIN_LOG_LEVEL=2 --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      --env OMP_NUM_THREADS=8 --env OPENBLAS_NUM_THREADS=1 --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?}" \
      "$software/containers/tensorflow-25.02.sif" "$software/tf2502/bin/python" "$@"
}
IFS=',' read -r -a models <<< "$models_csv"
index=${SLURM_ARRAY_TASK_ID:?};[[ "$index" =~ ^[0-9]+$ && "$index" -lt "${#models[@]}" ]];model=${models[$index]}
if [[ "$mode" == probe ]];then
    run_python tools/lowres/verify_geometry.py --final-sl --model "$model" --data "$data" \
      --manifests "$root/manifests" --source "$root/source/$model/fc2" --out "$root/probe/$model" --code-commit "$commit" \
      --final-sl-steps "$steps" --final-sl-initial-lr "$initial_lr"
elif [[ "$mode" == train ]];then
    python3 - "$root/probe/$model/acceptance.json" "$commit" "$steps" "$initial_lr" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]));assert a['passed'] and a['source_unchanged'] and a['code_commit']==sys.argv[2]
assert a['planned_steps']==int(sys.argv[3]) and a['full_sintel_best_selection'] and a['continuous_resumed_all_variables_exact']
assert a['initial_lr']==float(sys.argv[4]) and a['min_lr']==1e-6
assert a['actual_gpu_backprop'] and a['adam_reset_at_stage_start']
PY
    out="$root/seed42/mixture75_25/$model/replay"
    if [[ -f "$out/status.json" ]] && python3 - "$out/status.json" "$steps" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]));sys.exit(0 if a['completed'] and a['step']==int(sys.argv[2]) and a['total_steps']==int(sys.argv[2]) else 1)
PY
    then printf 'Fixed%s budget already completed.\n' "$steps";exit 0;fi
    if [[ -n "${5:-}" ]];then
        [[ "$5" =~ ^[0-9]+$ ]];previous="${5}_${index}"
        state=$(sacct --clusters="$SLURM_CLUSTER_NAME" -X -nP -j "$previous" -o JobID,State | awk -F'|' -v id="$previous" '$1==id {print $2}')
        case "$state" in TIMEOUT*|NODE_FAIL*|PREEMPTED*) ;; *) printf 'No retry: %s %s\n' "$previous" "$state";exit 2;; esac
    fi
    args=(--model "$model" --geometry random --phase fc2 --fc2-source-step 10000 --replay-arm mixture75_25 \
      --select-full-sintel --data "$data" --manifests "$root/manifests" --source "$root/source/$model/fc2" \
      --out "$out" --steps "$steps" --eval-every 1000 --initial-lr "$initial_lr" --min-lr 1e-6 --seed 42 --workers 8 --code-commit "$commit")
    [[ ! -f "$out/current.json" ]] || args+=(--resume)
    run_python tools/lowres/geometry_compare.py "${args[@]}"
else printf 'Unknown mode: %s\n' "$mode" >&2;exit 2;fi
