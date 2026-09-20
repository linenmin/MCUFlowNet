#!/bin/bash -l
set -euo pipefail
root=/data/leuven/379/vsc37996/MCUFlowNet-component
code=$root/releases/680eba5c21a234c6fffbf8fe77082433b3d87435
export APPTAINER_TMPDIR=${TMPDIR:-/tmp}/mcuflow-${SLURM_JOB_ID}
mkdir -p "$APPTAINER_TMPDIR"
image=$root/containers/tensorflow-25.02.sif
if [ ! -e "$image" ]; then
 apptainer pull --disable-cache "$image.partial-${SLURM_JOB_ID}" docker://nvcr.io/nvidia/tensorflow@sha256:c83b37d26f19ab00d8a13cf974edd079c3d099918ec3110c304a989d6e2f75d5
 mv -n "$image.partial-${SLURM_JOB_ID}" "$image"
fi
sha256sum "$image" > "$root/control/image.sha256"
environment=$root/environments/tf2502-v2
runner=(apptainer exec --cleanenv --bind "$root:$root" --bind /scratch/leuven/379/vsc37996/dataset:/datasets:ro --bind "$code:/workspace:ro" --pwd /workspace --env TF_USE_LEGACY_KERAS=1 --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES=-1 "$image")
if [ ! -f "$environment/READY" ]; then
 test ! -e "$environment"
 "${runner[@]}" python -m venv --without-pip --system-site-packages "$environment"
 printf 'numpy==1.26.4\nscipy==1.12.0\n' > "$root/control/constraints.txt"
 "${runner[@]}" "$environment/bin/python" -m pip install --no-cache-dir -c "$root/control/constraints.txt" opencv-python-headless==4.11.0.86 scikit-image==0.24.0 matplotlib==3.9.2 pillow==11.1.0
 "${runner[@]}" "$environment/bin/python" -m pip check
 "${runner[@]}" "$environment/bin/python" -m pip freeze > "$root/control/pip-freeze.txt"
 touch "$environment/READY"
fi
"${runner[@]}" "$environment/bin/python" "$root/control/tier2-data-audit.py"
touch "$root/control/DATA_READY"
