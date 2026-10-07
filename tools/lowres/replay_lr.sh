#!/bin/bash -l
set -euo pipefail
mode=$1 repo=$2 data=$3 root=$4
[[ "${SLURM_CLUSTER_NAME:?}" == mindwell ]]
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
index=${SLURM_ARRAY_TASK_ID:?};models=(edge S L edge S L)
if [[ "$mode" == probe ]];then
    [[ "$index" =~ ^[0-2]$ ]];model=${models[$index]}
    run_python tools/lowres/verify_replay_lr.py --model "$model" --data "$data" --manifests "$root/manifests" \
        --source "$root/source/$model/replay" --baseline-score "$root/source/$model/full-score.json" \
        --out "$root/probe/$model" --code-commit "$commit"
elif [[ "$mode" == train ]];then
    [[ "$index" =~ ^[0-5]$ ]];model=${models[$index]};policies=(fixed fixed fixed restart restart restart);policy=${policies[$index]}
    python3 - "$root/probe/$model/acceptance.json" "$commit" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]));assert a['passed'] and a['source_unchanged'] and a['paired_order_and_geometry_exact'] and a['code_commit']==sys.argv[2]
PY
    out="$root/seed42/$policy/$model/replay"
    if [[ -f "$out/status.json" ]] && python3 - "$out/status.json" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]));sys.exit(0 if a['stage_completed'] and a['phase_step']==5000 and a['step']==15000 else 1)
PY
    then printf 'Approved5000 phase updates complete; no automatic extension.\n';exit 0;fi
    if [[ -n "${5:-}" ]];then
        previous="${5}_${index}";[[ "$5" =~ ^[0-9]+$ ]]
        state=$(sacct --clusters=mindwell -X -nP -j "$previous" -o JobID,State | awk -F'|' -v id="$previous" '$1==id {print $2}')
        case "$state" in TIMEOUT*|NODE_FAIL*|PREEMPTED*) ;; *) printf 'No retry for %s %s\n' "$previous" "$state";exit 2;; esac
    fi
    args=(--model "$model" --policy "$policy" --data "$data" --manifests "$root/manifests" --source "$root/source/$model/replay" \
        --baseline-score "$root/source/$model/full-score.json" --out "$out" --phase-steps 10000 --stop-after 5000 \
        --eval-every 1000 --workers 8 --code-commit "$commit")
    [[ ! -f "$out/current.json" ]] || args+=(--resume)
    run_python tools/lowres/replay_lr_compare.py "${args[@]}"
else printf 'Unknown mode: %s\n' "$mode" >&2;exit 2;fi
