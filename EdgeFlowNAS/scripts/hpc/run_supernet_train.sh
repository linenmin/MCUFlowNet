#!/usr/bin/env bash
# HPC supernet training launcher

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIG_PATH="${CONFIG_PATH:-configs/supernet_fc2_180x240.yaml}"
GPU_DEVICE="${GPU_DEVICE:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
RESUME_EXPERIMENT_NAME="${RESUME_EXPERIMENT_NAME:-}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-0}"
DRY_RUN="${DRY_RUN:-0}"

cd "${PROJECT_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  fi
fi

CMD=("${PYTHON_BIN}" wrappers/run_supernet_train.py)
CMD+=(--config "${CONFIG_PATH}")
CMD+=(--gpu_device "${GPU_DEVICE}")
CMD+=(--num_epochs "${NUM_EPOCHS}")
CMD+=(--steps_per_epoch "${STEPS_PER_EPOCH}")
CMD+=(--batch_size "${BATCH_SIZE}")
CMD+=(--lr "${LEARNING_RATE}")

if [[ -n "${EXPERIMENT_NAME}" ]]; then
  CMD+=(--experiment_name "${EXPERIMENT_NAME}")
fi
if [[ -n "${RESUME_EXPERIMENT_NAME}" ]]; then
  CMD+=(--resume_experiment_name "${RESUME_EXPERIMENT_NAME}")
fi
if [[ "${LOAD_CHECKPOINT}" == "1" ]]; then
  CMD+=(--load_checkpoint)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=(--dry_run)
fi

echo "[EdgeFlowNAS][HPC] Running command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
