#!/bin/bash -l
# A bounded six-arm pilot. No automatic scientific continuation beyond 5000.
set -euo pipefail
mode=$1 repo=$2 data=$3 root=$4
[[ "${SLURM_CLUSTER_NAME:?}" == mindwell ]]
project="${root%/runs/LOWRES-BENCH-01/*}"
[[ "$project" != "$root" && "$project" = /* && "$(realpath "$project")" == "$project" ]]
software="$project/software"
reference_repo=$(cat "$root/control/reference-repo.txt")
reference_experiment="$project/runs/LOWRES-BENCH-01/ft3d-geometry10k-20261004"
[[ -f "$software/READY" && -f "$root/control/SOURCES_READY.json" ]]
cd "$repo"
commit=$(git rev-parse HEAD)
[[ "$commit" == "$(cat "$root/control/code-commit.txt")" && -z "$(git status --porcelain)" ]]
run_python() {
    apptainer exec --nv --cleanenv \
        --bind "$software:$software" --bind "$repo:$repo:ro" --bind "$reference_repo:$reference_repo:ro" --bind "$data:$data:ro" \
        --bind "$root:$root" --bind "$root/source:$root/source:ro" --bind "$reference_experiment:$reference_experiment:ro" --pwd "$repo" \
        --env TF_USE_LEGACY_KERAS=1 --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 \
        --env PYTHONUNBUFFERED=1 --env TF_DETERMINISTIC_OPS=1 --env TF_CPP_MIN_LOG_LEVEL=2 \
        --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env OMP_NUM_THREADS=8 --env OPENBLAS_NUM_THREADS=1 \
        --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?}" \
        "$software/containers/tensorflow-25.02.sif" "$software/tf2502/bin/python" "$@"
}
index=${SLURM_ARRAY_TASK_ID:?}
models=(edge S L edge S L)
if [[ "$mode" == probe ]]; then
    [[ "$index" =~ ^[0-2]$ ]];model=${models[$index]}
    run_python tools/lowres/verify_geometry.py --replay-compare --model "$model" --data "$data" \
        --manifests "$root/manifests" --source "$root/source/$model/fc2" \
        --out "$root/probe/$model" --code-commit "$commit" --reference-repo "$reference_repo"
elif [[ "$mode" == reference ]];then
    [[ "$index" =~ ^[0-2]$ ]];model=${models[$index]}
    python3 - "$root/probe/$model/acceptance.json" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]));assert a['passed'] and a['pure_ft_regression_all_variables_exact']
PY
    run_python tools/lowres/score_replay_reference.py --model "$model" --data "$data" \
        --manifests "$root/manifests" --experiment "$reference_experiment" \
        --out "$root/reference/$model" --code-commit "$commit"
elif [[ "$mode" == train ]]; then
    [[ "$index" =~ ^[0-5]$ ]];model=${models[$index]}
    arms=(fc2_only fc2_only fc2_only mixture75_25 mixture75_25 mixture75_25);arm=${arms[$index]}
    python3 - "$root/probe/$model/acceptance.json" "$commit" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]));assert a['passed'] and a['source_unchanged'] and a['code_commit']==sys.argv[2]
assert a['pure_ft_regression_all_variables_exact'] and a['tensorflow']=='2.17.0'
PY
    out="$root/seed42/$arm/$model/replay"
    if [[ -f "$out/status.json" ]] && python3 -c 'import json,sys;a=json.load(open(sys.argv[1]));assert a["pilot_completed"] and a["step"]==5000 and a["total_steps"]==10000' "$out/status.json";then
        printf 'Pilot already complete; no automatic continuation.\n';exit 0
    fi
    if [[ -n "${5:-}" ]];then
        [[ "$5" =~ ^[0-9]+$ ]];previous="${5}_${index}"
        state=$(sacct --clusters=mindwell -X -nP -j "$previous" -o JobID,State | awk -F'|' -v id="$previous" '$1==id {print $2}')
        case "$state" in TIMEOUT*|NODE_FAIL*|PREEMPTED*) ;; *) printf 'No resource retry: %s %s\n' "$previous" "$state";exit 2;; esac
    fi
    args=(--model "$model" --geometry random --phase fc2 --fc2-source-step 10000 \
        --replay-arm "$arm" --data "$data" --manifests "$root/manifests" \
        --source "$root/source/$model/fc2" --out "$out" --steps 10000 --stop-after-steps 5000 \
        --eval-every 1000 --initial-lr 3e-6 --min-lr 1e-6 --seed 42 --workers 8 --code-commit "$commit")
    [[ ! -f "$out/current.json" ]] || args+=(--resume)
    run_python tools/lowres/geometry_compare.py "${args[@]}"
else
    printf 'Unknown replay mode: %s\n' "$mode" >&2;exit 2
fi
