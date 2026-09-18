#!/bin/bash -l
# Submit with an explicit Sofia account, H200 partition, 1 GPU and 24 CPUs.
set -euo pipefail
project=$(realpath "$1")
code=$(realpath "$2")
case "$project" in /sofia/projects/*) ;; *) exit 2;; esac
root="$project/mcuflownet"
mkdir -p "$root/containers" "$root/environments" "$root/bootstrap"
exec > >(tee "$root/bootstrap/job-${SLURM_JOB_ID}.log") 2>&1
printf 'PROJECT=%s\nCODE=%s\nJOB=%s\n' "$project" "$code" "$SLURM_JOB_ID"
hostname
nvidia-smi
module -t list 2>&1 || true
image="$root/containers/tensorflow-25.02.sif"
export APPTAINER_TMPDIR="${TMPDIR:-/tmp}/mcuflow-${SLURM_JOB_ID}"
mkdir -p "$APPTAINER_TMPDIR"
if [ ! -f "$image" ]; then
    apptainer pull --disable-cache "$image.partial-${SLURM_JOB_ID}" \
      docker://nvcr.io/nvidia/tensorflow@sha256:c83b37d26f19ab00d8a13cf974edd079c3d099918ec3110c304a989d6e2f75d5
    mv -n "$image.partial-${SLURM_JOB_ID}" "$image"
fi
sha256sum "$image" | tee "$root/bootstrap/image.sha256"
environment="$root/environments/tf2502-v2"
runner=(apptainer exec --nv --cleanenv --bind "$project:$project" --bind "$code:/workspace:ro" --pwd /workspace
        --env TF_USE_LEGACY_KERAS=1 --env PYTHONDONTWRITEBYTECODE=1
        --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?Slurm GPU assignment missing}"
        --env TF_NUM_INTRAOP_THREADS=12 --env TF_NUM_INTEROP_THREADS=2 "$image")
if [ ! -f "$environment/READY" ]; then
    if [ -e "$environment" ]; then echo 'Unfinished environment exists; inspect it before retrying.'; exit 3; fi
    "${runner[@]}" python -m venv --without-pip --system-site-packages "$environment"
    printf 'numpy==1.26.4\nscipy==1.12.0\n' > "$root/bootstrap/constraints.txt"
    "${runner[@]}" "$environment/bin/python" -m pip install --no-cache-dir -c "$root/bootstrap/constraints.txt" \
      opencv-python-headless==4.11.0.86 scikit-image==0.24.0 matplotlib==3.9.2 pillow==11.1.0
    "${runner[@]}" "$environment/bin/python" -m pip check
    "${runner[@]}" "$environment/bin/python" -m pip freeze > "$root/bootstrap/pip-freeze.txt"
    touch "$environment/READY"
fi
"${runner[@]}" "$environment/bin/python" tools/setup/check_tensorflow.py --device gpu --output "$root/bootstrap/gpu-check.json"
"${runner[@]}" "$environment/bin/python" tools/hpc/audit_datasets.py --root "$project/datasets" --output "$root/bootstrap/datasets.json"
printf 'SUCCESS\n' > "$root/bootstrap/COMPLETE"
