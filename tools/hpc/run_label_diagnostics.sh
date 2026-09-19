#!/bin/bash -l
set -euo pipefail
project=$(realpath "$1")
code=$(realpath "$2")
campaign="$3"
variant="$4"
evaluation="$5"
root="$project/mcuflownet"
commit=$(git -C "$code" rev-parse HEAD)
test -z "$(git -C "$code" status --porcelain)"
printf 'JOB=%s\nCOMMIT=%s\nOUTPUT=%s\n' "$SLURM_JOB_ID" "$commit" "$root/runs/$evaluation/$variant"
apptainer exec --nv --cleanenv \
  --bind "$project:$project" --bind "$project/datasets:/datasets:ro" \
  --bind "$code:/workspace:ro" --pwd /workspace \
  --env TF_USE_LEGACY_KERAS=1 --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 \
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?}" --env "SLURM_JOB_ID=$SLURM_JOB_ID" \
  --env "MCUFLOW_COMMIT=$commit" --env TF_NUM_INTRAOP_THREADS=12 --env TF_NUM_INTEROP_THREADS=2 \
  "$root/containers/tensorflow-25.02.sif" "$root/environments/tf2502-v2/bin/python" \
  tools/validation/evaluate_label_diagnostics.py --runs "$root/runs" --campaign "$campaign" \
  --variant "$variant" --output "$root/runs/$evaluation/$variant"
