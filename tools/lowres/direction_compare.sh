#!/bin/bash -l
# Six paired FC2 fine-tunes from fixed random-geometry step10k sources.
set -euo pipefail
mode=$1 repo=$2 data=$3 root=$4
cluster=${SLURM_CLUSTER_NAME:?}
case "$cluster" in
    mindwell|wice) software="${VSC_SCRATCH_GPFS1:?}/MCUFlowNet/software"; environment="$software/tf2502"; ready="$software/READY" ;;
    *) printf 'Unverified training cluster: %s\n' "$cluster" >&2; exit 2 ;;
esac
[[ -f "$ready" && -f "$root/control/SOURCES_READY.json" ]]
cd "$repo"
commit=$(git rev-parse HEAD)
[[ "$commit" == "$(cat "$root/control/code-commit.txt")" && -z "$(git status --porcelain)" ]]
run_python() {
    apptainer exec --nv --cleanenv \
        --bind "$software:$software" --bind "$repo:$repo:ro" --bind "$data:$data:ro" \
        --bind "$root:$root" --bind "$root/source:$root/source:ro" --pwd "$repo" \
        --env TF_USE_LEGACY_KERAS=1 --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 \
        --env PYTHONUNBUFFERED=1 --env TF_DETERMINISTIC_OPS=1 --env TF_CPP_MIN_LOG_LEVEL=2 \
        --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env OMP_NUM_THREADS=8 --env OPENBLAS_NUM_THREADS=1 \
        --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?}" \
        "$software/containers/tensorflow-25.02.sif" "$environment/bin/python" "$@"
}
index=${SLURM_ARRAY_TASK_ID:?}
if [[ "$mode" == probe ]]; then
    models=(edge S L); [[ "$index" =~ ^[0-2]$ ]]; model=${models[$index]}
    run_python tools/lowres/test_direction_loss.py
    run_python tools/lowres/verify_geometry.py --model "$model" --data "$data" \
        --manifests "$root/manifests" --source "$root/source/$model/fc2" \
        --out "$root/probe/$model" --phase fc2 --direction-compare --code-commit "$commit"
elif [[ "$mode" == train ]]; then
    models=(edge S L edge S L); arms=(original original original weighted weighted weighted)
    [[ "$index" =~ ^[0-5]$ ]]; model=${models[$index]}; arm=${arms[$index]}
    run_python -c 'import json,sys; a=json.load(open(sys.argv[1])); assert a["passed"] and a["source_unchanged"] and a["code_commit"]==sys.argv[2] and a["direction_comparison"] and a["paired_geometry_equal"]' \
        "$root/probe/$model/acceptance.json" "$commit"
    weights=(1 1); [[ "$arm" != weighted ]] || weights=(1.30879345603272 0.6912065439672802)
    out="$root/seed42/$arm/$model/fc2"
    if [[ -f "$out/status.json" ]] && python3 -c 'import json,sys; a=json.load(open(sys.argv[1])); assert a["completed"] and a["step"]==a["total_steps"]==5000' "$out/status.json"; then
        printf 'All 5000 updates already complete.\n'; exit 0
    fi
    args=(--model "$model" --geometry random --phase fc2 --fc2-source-step 10000 \
        --direction-weights "${weights[@]}" --data "$data" --manifests "$root/manifests" \
        --source "$root/source/$model/fc2" --out "$out" --steps 5000 --eval-every 1000 \
        --initial-lr 3e-6 --min-lr 1e-6 --seed 42 --workers 8 --code-commit "$commit")
    [[ ! -f "$out/current.json" ]] || args+=(--resume)
    run_python tools/lowres/geometry_compare.py "${args[@]}"
else
    printf 'Unknown mode: %s\n' "$mode" >&2; exit 2
fi
