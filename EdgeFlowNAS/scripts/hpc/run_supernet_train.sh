#!/usr/bin/env bash
# HPC 超网训练模板脚本

set -euo pipefail  # 开启严格模式

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # 计算项目根目录
cd "${PROJECT_ROOT}"  # 切换到项目根目录

python wrappers/run_supernet_train.py \  # 调用超网训练入口
  --config configs/supernet_fc2_180x240.yaml \  # 指定配置文件
  --gpu_device 0 \  # 指定GPU编号
  --num_epochs 200 \  # 指定训练轮数
  --batch_size 32 \  # 指定批大小
  --lr 1e-4  # 指定学习率

