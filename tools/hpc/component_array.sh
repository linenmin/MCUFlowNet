#!/bin/bash -l
set -euo pipefail
project="$1"
code="$2"
mode="$3"
exec bash "$code/tools/hpc/run_sofia_task.sh" "$project" "$code" \
  tools/hpc/run_component_ablation.py --mode "$mode" --index "${SLURM_ARRAY_TASK_ID:?}"
