#!/bin/bash -l
set -euo pipefail
[[ "${SLURM_CLUSTER_NAME:-}" == mindwell ]]
mode=$1 repo=$2 data=$3 root=$4
software="$VSC_SCRATCH/MCUFlowNet/software"
[[ -f "$software/READY" && -f "$root/control/DATA_READY.json" ]]
export LOWRES_RUNTIME_READY=1
python() {
    apptainer exec --nv --cleanenv \
        --bind "$software:$software" --bind "$repo:$repo:ro" \
        --bind "$data:$data:ro" --bind "$root:$root" --pwd "$repo" \
        --env TF_USE_LEGACY_KERAS=1 --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 \
        --env PYTHONUNBUFFERED=1 --env TF_DETERMINISTIC_OPS=1 \
        --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env OMP_NUM_THREADS=8 --env OPENBLAS_NUM_THREADS=1 \
        --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?}" \
        "$software/containers/tensorflow-25.02.sif" "$software/tf2502/bin/python" "$@"
}
# Checkpoint provenance is supplied by the host because a Git worktree's metadata
# directory is not necessarily available inside the read-only container mount.
export LOWRES_HOST_COMMIT=$(git -C "$repo" rev-parse HEAD)
source "$repo/tools/lowres/factor_compare.sh" "$@"
