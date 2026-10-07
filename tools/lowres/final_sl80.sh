#!/bin/bash -l
set -euo pipefail
mode=$1 repo=$2 data=$3 root=$4
[[ "${SLURM_CLUSTER_NAME:?}" == mindwell || "$SLURM_CLUSTER_NAME" == wice ]]
project="${root%/runs/LOWRES-BENCH-01/*}";software="$project/software"
[[ "$project" != "$root" && "$(realpath "$project")" == "$project" ]]
[[ -f "$software/READY" && -f "$root/control/READY.json" ]]
cd "$repo";commit=$(git rev-parse HEAD)
[[ "$commit" == "$(cat "$root/control/code-commit.txt")" && -z "$(git status --porcelain)" ]]
index=${SLURM_ARRAY_TASK_ID:?};[[ "$index" =~ ^[01]$ ]];models=(S L);model=${models[$index]}
run_python() {
    apptainer exec --nv --cleanenv --bind "$software:$software" --bind "$repo:$repo:ro" --bind "$data:$data:ro" \
      --bind "$root:$root" --bind "$root/source:$root/source:ro" --pwd "$repo" \
      --env TF_USE_LEGACY_KERAS=1 --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 \
      --env TF_DETERMINISTIC_OPS=1 --env TF_CPP_MIN_LOG_LEVEL=2 --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      --env OMP_NUM_THREADS=8 --env OPENBLAS_NUM_THREADS=1 --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?}" \
      "$software/containers/tensorflow-25.02.sif" "$software/tf2502/bin/python" "$@"
}
common=(--final-sl80 --model "$model" --data "$data" --manifests "$root/manifests" \
        --source "$root/source/$model/replay" --baseline-score "$root/control/source-40k-checkpoints-verified.json" --code-commit "$commit")
if [[ "$mode" == probe ]];then
    run_python tools/lowres/verify_replay_lr.py "${common[@]}" --out "$root/probe/$model"
elif [[ "$mode" == train ]];then
    python3 - "$root/probe/$model/acceptance.json" "$commit" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]));assert a['passed'] and a['source_unchanged'] and a['code_commit']==sys.argv[2]
assert a['source_global_step']==40000 and a['planned_end_step']==80000 and a['final_sl80']
assert all(a[k] for k in ('initial_all_variables_exact','adam_preserved_nonzero','continuous_resumed_all_variables_exact','actual_gpu_backprop','learning_rate_fixed1e6'))
PY
    out="$root/seed42/mixture75_25/$model/replay"
    if [[ -f "$out/status.json" ]] && python3 - "$out/status.json" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]));sys.exit(0 if a['completed'] and a['step']==80000 else 1)
PY
    then printf 'Fixed80000 budget already completed.\n';exit 0;fi
    if [[ -n "${5:-}" ]];then
        [[ "$5" =~ ^[0-9]+$ ]];previous="${5}_${index}"
        state=$(sacct --clusters="$SLURM_CLUSTER_NAME" -X -nP -j "$previous" -o JobID,State | awk -F'|' -v id="$previous" '$1==id {print $2}')
        case "$state" in TIMEOUT*|NODE_FAIL*|PREEMPTED*) ;; *) printf 'No retry: %s %s\n' "$previous" "$state";exit 2;; esac
    fi
    args=("${common[@]}" --policy fixed --phase-steps 40000 --stop-after 40000 --eval-every 1000 --workers 8 --out "$out")
    [[ ! -f "$out/current.json" ]] || args+=(--resume)
    run_python tools/lowres/replay_lr_compare.py "${args[@]}"
else printf 'Unknown mode: %s\n' "$mode" >&2;exit 2;fi
