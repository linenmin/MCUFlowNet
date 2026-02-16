#!/usr/bin/env bash
# HPC 超网训练脚本模板

set -euo pipefail  # 开启严格模式

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # 计算项目根目录
CONFIG_PATH="${CONFIG_PATH:-configs/supernet_fc2_180x240.yaml}"  # 读取配置路径环境变量
GPU_DEVICE="${GPU_DEVICE:-0}"  # 读取GPU编号环境变量
NUM_EPOCHS="${NUM_EPOCHS:-200}"  # 读取训练轮数环境变量
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-50}"  # 读取每轮步数环境变量
BATCH_SIZE="${BATCH_SIZE:-32}"  # 读取批大小环境变量
LEARNING_RATE="${LEARNING_RATE:-1e-4}"  # 读取学习率环境变量
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"  # 读取实验名环境变量
RESUME_EXPERIMENT_NAME="${RESUME_EXPERIMENT_NAME:-}"  # 读取恢复实验名环境变量
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-0}"  # 读取是否加载断点环境变量
FAST_MODE="${FAST_MODE:-0}"  # 读取快速模式环境变量

cd "${PROJECT_ROOT}"  # 切换到项目根目录

CMD=(python wrappers/run_supernet_train.py)  # 初始化训练命令数组
CMD+=(--config "${CONFIG_PATH}")  # 添加配置文件参数
CMD+=(--gpu_device "${GPU_DEVICE}")  # 添加GPU参数
CMD+=(--num_epochs "${NUM_EPOCHS}")  # 添加轮数参数
CMD+=(--steps_per_epoch "${STEPS_PER_EPOCH}")  # 添加每轮步数参数
CMD+=(--batch_size "${BATCH_SIZE}")  # 添加批大小参数
CMD+=(--lr "${LEARNING_RATE}")  # 添加学习率参数

if [[ -n "${EXPERIMENT_NAME}" ]]; then  # 判断是否显式指定实验名
  CMD+=(--experiment_name "${EXPERIMENT_NAME}")  # 追加实验名参数
fi  # 结束实验名分支

if [[ -n "${RESUME_EXPERIMENT_NAME}" ]]; then  # 判断是否显式指定恢复实验名
  CMD+=(--resume_experiment_name "${RESUME_EXPERIMENT_NAME}")  # 追加恢复实验名参数
fi  # 结束恢复实验名分支

if [[ "${LOAD_CHECKPOINT}" == "1" ]]; then  # 判断是否启用断点恢复
  CMD+=(--load_checkpoint)  # 追加断点恢复参数
fi  # 结束断点恢复分支

if [[ "${FAST_MODE}" == "1" ]]; then  # 判断是否启用快速模式
  CMD+=(--fast_mode)  # 追加快速模式参数
fi  # 结束快速模式分支

echo "[EdgeFlowNAS][HPC] Running command:"  # 输出命令提示
printf ' %q' "${CMD[@]}"  # 按可复制格式打印命令
echo  # 输出换行

"${CMD[@]}"  # 执行训练命令
