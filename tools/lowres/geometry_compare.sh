#!/bin/bash -l
# Common scratch-initialized FC2 experiment; six paired geometry arms.
set -euo pipefail
mode=$1 repo=$2 data=$3 root=$4 phase=${6:-fc2}
[[ "$phase" == fc2 || "$phase" == ft3d ]]
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
        --out "$root/probe/$model" --phase "$phase" --code-commit "$commit"
elif [[ "$mode" == train ]]; then
    models=(edge S L edge S L); arms=(whole whole whole random random random)
    [[ "$index" =~ ^[0-5]$ ]]
    model=${models[$index]} arm=${arms[$index]}
    run_python -c 'import json,sys; a=json.load(open(sys.argv[1])); assert a["passed"] and a["source_unchanged"] and a["code_commit"]==sys.argv[2] and a.get("phase","fc2")==sys.argv[3]' \
        "$root/probe/$model/acceptance.json" "$commit" "$phase"
    out="$root/seed42/$arm/$model/$phase"
    if [[ -n "${5:-}" ]]; then
        if [[ -f "$out/status.json" ]] && python3 -c 'import json,sys; a=json.load(open(sys.argv[1])); assert a["completed"] and a["step"]==10000' "$out/status.json"; then
            printf 'All 10000 steps complete; no training needed.\n'; exit 0
        fi
        previous="${5}_${index}"
        # JobIDRaw is the internal child ID, not ARRAY_ID_TASK_ID.
        predecessor_state=''
        for attempt in {1..15}; do
            predecessor_state=$(sacct --clusters=mindwell -X -nP -j "$previous" -o JobID,State | awk -F'|' -v id="$previous" '$1==id {print $2}')
            case "$predecessor_state" in
                ''|RUNNING*|PENDING*|COMPLETING*) sleep 2 ;;
                *) break ;;
            esac
        done
        case "$predecessor_state" in
            TIMEOUT*|NODE_FAIL*|PREEMPTED*) ;;
            *) printf 'No retry for %s state=%s\n' "$previous" "$predecessor_state"; exit 2 ;;
        esac
        # Preserve a first-boundary interruption instead of overwriting it.
        python3 - "$root" "$out" "$previous" "$phase" <<'PY'
from pathlib import Path
import sys
r=Path(sys.argv[1]).resolve(); p=Path(sys.argv[2]); predecessor=sys.argv[3]; phase=sys.argv[4]
assert phase in ('fc2','ft3d') and not p.is_symlink() and p.resolve().is_relative_to(r/'seed42') and p.name==phase
if p.exists() and not (p/'current.json').exists():
    saved=p.with_name(phase+'.incomplete-after-'+predecessor)
    assert not saved.exists() and saved.resolve().is_relative_to(r/'seed42')
    print('Preserved interrupted first boundary:',p.resolve(),'->',saved.resolve(),flush=True)
    p.rename(saved)
PY
    fi
    args=(--model "$model" --geometry "$arm" --data "$data" --manifests "$root/manifests" \
          --source "$root/source/$model/fc2" --out "$out" --steps 10000 --eval-every 1000 --seed 42 --phase "$phase")
    [[ "$phase" != ft3d ]] || args+=(--initial-lr 3e-6 --min-lr 1e-6)
    [[ ! -f "$out/current.json" ]] || args+=(--resume)
    run_python tools/lowres/geometry_compare.py "${args[@]}" --code-commit "$commit"
else
    printf 'Unknown mode: %s\n' "$mode" >&2; exit 2
fi
