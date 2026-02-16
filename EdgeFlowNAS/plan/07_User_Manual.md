# EdgeFlowNAS Supernet 使用手册

更新时间: 2026-02-16

## 1. 文档目标

本手册用于指导你在 `MCUFlowNet/EdgeFlowNAS` 中完成以下工作:

1. 本机环境下启动 Supernet 训练。
2. 在训练后执行固定子网池评估。
3. 进行断点恢复训练。
4. 验收训练产物完整性。
5. 在 HPC 场景下复用脚本模板。

当前手册范围仅覆盖 **Supernet 训练阶段**，不包含 NAS 搜索与最终子网重训。

## 2. 项目路径与关键文件

项目根目录:

`MCUFlowNet/EdgeFlowNAS`

关键入口:

1. 训练入口: `MCUFlowNet/EdgeFlowNAS/wrappers/run_supernet_train.py`
2. 评估入口: `MCUFlowNet/EdgeFlowNAS/code/nas/supernet_eval.py`
3. 产物校验: `MCUFlowNet/EdgeFlowNAS/code/nas/check_manifest.py`
4. 默认配置: `MCUFlowNet/EdgeFlowNAS/configs/supernet_fc2_180x240.yaml`
5. HPC 模板: `MCUFlowNet/EdgeFlowNAS/scripts/hpc/run_supernet_train.sh`

## 3. 环境准备

### 3.1 本机训练验证环境

你当前约定的本机仿真环境:

`conda activate tf_work_hpc`

建议先做环境可用性检查:

```bash
conda run -n tf_work_hpc python --version
conda run -n tf_work_hpc python -c "import tensorflow as tf; print(tf.__version__)"
```

### 3.2 Bash 环境说明

`scripts/hpc/run_supernet_train.sh` 用于 Linux/HPC 场景。  
在 Windows 本机若通过 WSL/bash 调用，可能没有 TensorFlow 环境。  
手册给出两种策略:

1. 在本机先走 Python 直调训练命令。
2. 对 HPC 脚本在本机优先使用 `DRY_RUN=1` 验证参数组装。

## 4. 数据配置

默认配置文件已锁定:

`MCUFlowNet/EdgeFlowNAS/configs/supernet_fc2_180x240.yaml`

数据相关字段:

1. `data.dataset=FC2`
2. `data.data_list=../EdgeFlowNet/code/dataset_paths`
3. `data.base_path=../`
4. `data.train_list_name=FC2_train.txt`
5. `data.val_list_name=FC2_test.txt`
6. `data.input_height=180`
7. `data.input_width=240`

说明:

1. Supernet 阶段按你确认方案使用 `FC2_test` 作为排序验证来源。
2. 训练输入统一为 `180x240`。

## 5. 一次完整流程

## Step-1: 参数与配置检查

```bash
cd MCUFlowNet/EdgeFlowNAS
python wrappers/run_supernet_train.py --help
python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --dry_run
```

预期:

1. `--help` 正常输出完整参数列表。
2. `--dry_run` 输出合并后的配置 JSON，不进入训练图构建。

## Step-2: 启动训练

推荐先做小步数 smoke:

```bash
conda run -n tf_work_hpc python wrappers/run_supernet_train.py \
  --config configs/supernet_fc2_180x240.yaml \
  --experiment_name edgeflownas_supernet_fc2_180x240_smoke \
  --num_epochs 2 \
  --steps_per_epoch 1 \
  --batch_size 2 \
  --lr 1e-4 \
  --fast_mode
```

标准长跑可仅调整 `num_epochs/steps_per_epoch/batch_size`。

## Step-3: 断点恢复

```bash
conda run -n tf_work_hpc python wrappers/run_supernet_train.py \
  --config configs/supernet_fc2_180x240.yaml \
  --experiment_name edgeflownas_supernet_fc2_180x240_smoke \
  --resume_experiment_name edgeflownas_supernet_fc2_180x240_smoke \
  --load_checkpoint \
  --num_epochs 5 \
  --steps_per_epoch 1 \
  --batch_size 2 \
  --fast_mode
```

预期日志包含:

`resume checkpoint=... start_epoch=... global_step=...`

## Step-4: 固定子网池评估

```bash
conda run -n tf_work_hpc python -m code.nas.supernet_eval \
  --config configs/supernet_fc2_180x240.yaml \
  --eval_only \
  --checkpoint_type last \
  --bn_recal_batches 8 \
  --batch_size 2
```

输出文件:

`MCUFlowNet/EdgeFlowNAS/outputs/supernet/<experiment_name>/supernet_eval_result_last.json`

## Step-5: 产物验收

```bash
python -m code.nas.check_manifest \
  --path outputs/supernet/<experiment_name>/train_manifest.json \
  --strict
```

`ok=true` 时表示:

1. `manifest` 字段完整。
2. `fairness_counts` 三选项计数一致。
3. `eval_epe_history.csv` 核心列齐全且有数据行。

## 6. 输出目录说明

输出根目录:

`MCUFlowNet/EdgeFlowNAS/outputs/supernet/<experiment_name>`

核心产物:

1. `checkpoints/supernet_best.ckpt*`
2. `checkpoints/supernet_last.ckpt*`
3. `checkpoints/supernet_best.ckpt.meta.json`
4. `checkpoints/supernet_last.ckpt.meta.json`
5. `eval_pool_12.json`
6. `eval_epe_history.csv`
7. `fairness_counts.json`
8. `train_manifest.json`
9. `supernet_training_report.md`
10. `supernet_eval_result_{best|last}.json`
11. `train.log`

## 7. HPC 脚本用法

脚本:

`MCUFlowNet/EdgeFlowNAS/scripts/hpc/run_supernet_train.sh`

支持环境变量:

1. `PYTHON_BIN`
2. `CONFIG_PATH`
3. `GPU_DEVICE`
4. `NUM_EPOCHS`
5. `STEPS_PER_EPOCH`
6. `BATCH_SIZE`
7. `LEARNING_RATE`
8. `EXPERIMENT_NAME`
9. `RESUME_EXPERIMENT_NAME`
10. `LOAD_CHECKPOINT` (`0/1`)
11. `FAST_MODE` (`0/1`)
12. `DRY_RUN` (`0/1`)

本机参数组装验证示例:

```bash
cd /mnt/d/Dataset/MCUFlowNet/EdgeFlowNAS
NUM_EPOCHS=1 \
STEPS_PER_EPOCH=1 \
BATCH_SIZE=2 \
LEARNING_RATE=1e-4 \
FAST_MODE=1 \
DRY_RUN=1 \
EXPERIMENT_NAME=edgeflownas_supernet_fc2_180x240_hpc_smoke \
bash scripts/hpc/run_supernet_train.sh
```

HPC 实跑示例:

```bash
cd /path/to/MCUFlowNet/EdgeFlowNAS
NUM_EPOCHS=200 \
STEPS_PER_EPOCH=50 \
BATCH_SIZE=32 \
LEARNING_RATE=1e-4 \
FAST_MODE=0 \
EXPERIMENT_NAME=edgeflownas_supernet_fc2_180x240_run1 \
bash scripts/hpc/run_supernet_train.sh
```

## 8. 常见问题

1. 问题: `ModuleNotFoundError: tensorflow`  
处理: 切到 `tf_work_hpc` 或安装 TF1 兼容环境后再执行训练。

2. 问题: `checkpoint not found`  
处理: 检查 `--resume_experiment_name` 对应目录下是否存在 `checkpoints/supernet_last.ckpt.index`。

3. 问题: `check_manifest --strict` 失败  
处理: 逐项查看输出中的 `manifest/fairness_counts/eval_history` 子结果定位缺失文件或字段。

4. 问题: bash 下找不到 `python`  
处理: 设置 `PYTHON_BIN`，或在 Linux 环境安装 `python/python3` 并确保 PATH 可见。

## 9. 推荐执行顺序

1. `--help` 与 `--dry_run` 先通过。
2. 先跑 1-2 epoch smoke。
3. 再启长跑训练。
4. 每轮关键实验结束后跑 `supernet_eval`。
5. 用 `check_manifest --strict` 做产物验收。
6. 通过后再进入后续 NAS 搜索与子网重训阶段。
