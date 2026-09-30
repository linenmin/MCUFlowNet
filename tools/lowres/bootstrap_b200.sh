#!/bin/bash -l
# Run on a Mindwell CPU node; reproduce the existing local/Sofia runtime.
set -euo pipefail
[[ "${SLURM_CLUSTER_NAME:-}" == mindwell ]]
root="$VSC_SCRATCH/MCUFlowNet/software"
mkdir -p "$root/tmp" "$root/containers"
printf 'software=%s\n' "$(realpath "$root")"
export APPTAINER_TMPDIR="$root/tmp" APPTAINER_CACHEDIR="$root/cache"
image="$root/containers/tensorflow-25.02.sif"
if [[ ! -f "$image" ]]; then
    apptainer pull --disable-cache "$image.partial-$SLURM_JOB_ID" \
        docker://nvcr.io/nvidia/tensorflow@sha256:c83b37d26f19ab00d8a13cf974edd079c3d099918ec3110c304a989d6e2f75d5
    mv -n "$image.partial-$SLURM_JOB_ID" "$image"
fi
sha256sum "$image" > "$root/image.sha256"
envdir="$root/tf2502"
runner=(apptainer exec --cleanenv --bind "$root:$root" --env TF_USE_LEGACY_KERAS=1 "$image")
if [[ ! -f "$envdir/READY" ]]; then
    [[ ! -e "$envdir" ]] || { echo 'Inspect incomplete environment before retrying'; exit 2; }
    "${runner[@]}" python -m venv --without-pip --system-site-packages "$envdir"
    printf 'numpy==1.26.4\nscipy==1.12.0\n' > "$root/constraints.txt"
    "${runner[@]}" "$envdir/bin/python" -m pip install --no-cache-dir -c "$root/constraints.txt" \
        opencv-python-headless==4.11.0.86 scikit-image==0.24.0 matplotlib==3.9.2 pillow==11.1.0
    "${runner[@]}" "$envdir/bin/python" -m pip check
    "${runner[@]}" "$envdir/bin/python" -m pip freeze > "$root/pip-freeze.txt"
    touch "$envdir/READY"
fi
"${runner[@]}" "$envdir/bin/python" -c 'import tensorflow as tf,cv2,numpy; print(tf.__version__,cv2.__version__,numpy.__version__)'
touch "$root/READY"
