#!/bin/bash -l
set -euo pipefail
root=/data/leuven/379/vsc37996/MCUFlowNet-component
code=$root/releases/680eba5c21a234c6fffbf8fe77082433b3d87435
runs=/scratch/leuven/379/vsc37996/MCUFlowNet-component/runs
test -f "$root/control/DATA_READY"
test -z "$(git -C "$code" status --porcelain)"
mode=$1
stop_after=${2:-400}
index=${SLURM_ARRAY_TASK_ID:?}
export APPTAINER_TMPDIR=${TMPDIR:-/tmp}/mcuflow-${SLURM_JOB_ID}
mkdir -p "$APPTAINER_TMPDIR"
nvidia-smi
apptainer exec --nv --cleanenv --bind "$root:$root" --bind /scratch/leuven/379/vsc37996/dataset:/datasets:ro --bind "$runs:/runs" --bind "$code:/workspace:ro" --pwd /workspace \
 --env TF_USE_LEGACY_KERAS=1 --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 \
 --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?}" --env "SLURM_JOB_ID=$SLURM_JOB_ID" --env "SLURM_ARRAY_TASK_ID=$index" \
 --env MCUFLOW_COMMIT=680eba5c21a234c6fffbf8fe77082433b3d87435 --env TF_NUM_INTRAOP_THREADS=12 --env TF_NUM_INTEROP_THREADS=2 \
 "$root/containers/tensorflow-25.02.sif" "$root/environments/tf2502-v2/bin/python" tools/hpc/run_component_ablation.py --mode "$mode" --index "$index" --stop-after "$stop_after"
