#!/bin/bash -l
# Fixed-source, no-gradient diagnostic in the verified Sofia runtime.
set -euo pipefail
mode=$1 repo=$(realpath "$2") root=$(realpath "$3")
[[ "${SLURM_CLUSTER_NAME:?}" == sofia ]]
software="$HOME/Software/MCUFlowNet"
envdir="$software/environments/tf2502-v2"
[[ -f "$envdir/READY" && -f "$root/control/code-commit.txt" ]]
commit=$(git -C "$repo" rev-parse HEAD)
[[ "$commit" == "$(cat "$root/control/code-commit.txt")" ]]
[[ -z "$(git -C "$repo" status --porcelain)" ]]
run_python() {
    apptainer exec --nv --cleanenv \
        --bind "$software:$software:ro" --bind "$repo:/workspace:ro" \
        --bind "$root:/audit" --bind "$root/experiment-source:/audit/experiment-source:ro" \
        --bind /sofia/projects/2026_start_060/datasets:/datasets:ro --pwd /workspace \
        --env TF_USE_LEGACY_KERAS=1 --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 \
        --env TF_DETERMINISTIC_OPS=1 --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        --env PYTHONUNBUFFERED=1 --env OMP_NUM_THREADS=8 --env OPENBLAS_NUM_THREADS=1 \
        --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?}" \
        "$software/containers/tensorflow-25.02.sif" "$envdir/bin/python" "$@"
}
if [[ "$mode" == prepare ]]; then
    run_python tools/lowres/audit_domain_bn.py prepare --data /datasets \
        --experiment /audit/experiment-source --out /audit/prepared --code-commit "$commit"
else
    models=(edge S L)
    [[ "${SLURM_ARRAY_TASK_ID:?}" =~ ^[0-2]$ ]]
    model=${models[$SLURM_ARRAY_TASK_ID]}
    case "$mode" in
        probe)
            run_python tools/lowres/audit_domain_bn.py run --data /datasets \
                --prepared /audit/prepared --model "$model" --smoke \
                --out "/audit/probe/$model" --code-commit "$commit"
            ;;
        run)
            python3 - "$root/probe/$model/results.json" "$commit" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['completed'] and x['smoke_only'] and x['source_unchanged'] and x['code_commit']==sys.argv[2]
PY
            run_python tools/lowres/audit_domain_bn.py run --data /datasets \
                --prepared /audit/prepared --model "$model" \
                --out "/audit/results/$model" --code-commit "$commit"
            ;;
        *) printf 'Unknown diagnostic mode: %s\n' "$mode" >&2; exit 2 ;;
    esac
fi
