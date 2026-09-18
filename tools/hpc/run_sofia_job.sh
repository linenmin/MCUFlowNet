#!/bin/bash -l
set -euo pipefail
project=$(realpath "$1")
code=$(realpath "$2")
campaign="$3"
mode="$4"
variant="$5"
stop_after="${6:-15}"
root="$project/mcuflownet"
test -f "$root/bootstrap/COMPLETE"
test -f "$root/environments/tf2502-v2/READY"
mkdir -p "$root/runs"
export APPTAINER_TMPDIR="${TMPDIR:-/tmp}/mcuflow-${SLURM_JOB_ID}"
mkdir -p "$APPTAINER_TMPDIR"
commit=$(git -C "$code" rev-parse HEAD)
test -z "$(git -C "$code" status --porcelain)"
printf 'JOB=%s\nCOMMIT=%s\nCODE=%s\nCAMPAIGN=%s\nVARIANT=%s\n' "$SLURM_JOB_ID" "$commit" "$code" "$campaign" "$variant"
apptainer exec --nv --cleanenv \
  --bind "$project:$project" --bind "$project/datasets:$project/datasets:ro" \
  --bind "$project/datasets:/datasets:ro" --bind "$root/runs:/runs" --bind "$code:/workspace:ro" \
  --pwd /workspace --env TF_USE_LEGACY_KERAS=1 --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 \
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?}" --env "SLURM_JOB_ID=$SLURM_JOB_ID" \
  --env "MCUFLOW_COMMIT=$commit" --env TF_NUM_INTRAOP_THREADS=12 --env TF_NUM_INTEROP_THREADS=2 \
  "$root/containers/tensorflow-25.02.sif" "$root/environments/tf2502-v2/bin/python" \
  tools/hpc/run_campaign.py --campaign "$campaign" --mode "$mode" --variant "$variant" --stop-after "$stop_after"
