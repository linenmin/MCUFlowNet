# 04 超网训练执行门禁清单（仅 Supernet）

更新时间: 2026-02-16

## 1. 使用方式

本文件用于把 `03_Supernet_Task_Breakdown.md` 变成可执行顺序。  
执行时按 Gate-1 到 Gate-4 依次推进，前一 Gate 未通过不进入下一 Gate。
执行前先满足 `05_Engineering_File_Management_and_Code_Style.md` 的分层与注释规范。

## 2. Gate 总览

1. Gate-1: 基础工程可用（T01-T02）
2. Gate-2: 模型与公平采样可用（T03-T04）
3. Gate-3: 训练与评估闭环可用（T05-T07）
4. Gate-4: 交付与环境验收可用（T08-T09）

## 3. Gate-1 基础工程可用（T01-T02）

任务项：

1. 建立 `run_supernet_train.py` 参数入口。
2. 建立 `train_supernet.py` 主训练入口（含 dry-run）。
3. 完成 `arch_codec.py` 的 9 维解析与校验。

必须通过的命令：

1. `python wrappers/run_supernet_train.py --help`
2. `python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --dry_run`
3. `python -m code.nas.arch_codec --self_test`

通过标准：

1. 帮助信息覆盖核心参数：`gpu_device/num_epochs/batch_size/lr/config/supernet_mode`。
2. dry-run 可打印最终配置，不触发训练图构建。
3. `arch_code` 非法长度和非法取值可被阻断并返回明确错误信息。

产出物：

1. `EdgeFlowNAS/wrappers/run_supernet_train.py`
2. `EdgeFlowNAS/code/train_supernet.py`
3. `EdgeFlowNAS/code/nas/arch_codec.py`
4. `EdgeFlowNAS/configs/supernet_fc2_180x240.yaml`

## 4. Gate-2 模型与公平采样可用（T03-T04）

任务项：

1. 实现 `MultiScaleResNet_supernet.py`，支持 `arch_code` 动态切换。
2. 实现严格公平采样器 `fair_sampler.py`。

必须通过的命令：

1. `python -m code.nas.model_shape_check --h 180 --w 240 --batch 2 --samples 20`
2. `python -m code.nas.fair_sampler --cycles 300 --seed 42 --check`

通过标准：

1. 20 组随机 `arch_code` 前向全部成功。
2. 多尺度输出分辨率固定：`45x60`、`90x120`、`180x240`。
3. 每个 block 的三选项计数严格相等，`fairness_gap=0`。
4. 同 seed 下采样序列字节级一致。

产出物：

1. `EdgeFlowNAS/code/network/MultiScaleResNet_supernet.py`
2. `EdgeFlowNAS/code/nas/fair_sampler.py`
3. `EdgeFlowNAS/outputs/supernet/fair_sampler_check.json`

## 5. Gate-3 训练与评估闭环可用（T05-T07）

任务项：

1. 完成 3-path 同 batch 累积反向 + 单次优化更新。
2. 接入 `Adam + cosine decay + wd=4e-5 + clip=5.0`。
3. 完成固定 12 子网评估、BN 重估 8 batches、EPE 记录。
4. 完成早停、best/last checkpoint、断点续训。

必须通过的命令：

1. `python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --max_epochs 1 --max_steps 30 --debug_trace`
2. `python -m code.nas.supernet_eval --config configs/supernet_fc2_180x240.yaml --eval_only --bn_recal_batches 8`
3. `python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --max_epochs 3 && python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --resume --max_epochs 5`

通过标准：

1. debug 日志显示每公平周期 `forward/backward x3` 且 `optimizer_step x1`。
2. 梯度裁剪日志满足 `global_grad_norm_after <= 5.0`。
3. 每个 epoch 写入 `mean_epe_12/std_epe_12/fairness_gap`。
4. 每个子网评估日志包含 `bn_recal_batches=8`。
5. resume 后 `global_step` 连续，无重复和跳步。
6. best/last checkpoint 同时存在且指向正确 epoch。

产出物：

1. `EdgeFlowNAS/outputs/supernet/checkpoints/supernet_best.ckpt`
2. `EdgeFlowNAS/outputs/supernet/checkpoints/supernet_last.ckpt`
3. `EdgeFlowNAS/outputs/supernet/eval_pool_12.json`
4. `EdgeFlowNAS/outputs/supernet/eval_epe_history.csv`

## 6. Gate-4 交付与环境验收可用（T08-T09）

任务项：

1. 生成完整可复现实验清单（manifest + fairness + eval history）。
2. 在本机 `tf_work_hpc` 完成 2-epoch smoke。
3. 输出 HPC 命令模板（仅替换路径与资源参数）。

必须通过的命令：

1. `conda activate tf_work_hpc && python wrappers/run_supernet_train.py --gpu_device 0 --num_epochs 2 --batch_size 32 --lr 1e-4 --config configs/supernet_fc2_180x240.yaml --fast_mode`
2. `python -m code.nas.check_manifest --path outputs/supernet/train_manifest.json`

通过标准：

1. smoke 运行完成 2 epochs 并生成 checkpoint 与评估日志。
2. `train_manifest.json` 包含字段：`seed/config_hash/git_commit/input_shape/optimizer/lr_schedule/wd/grad_clip`。
3. `fairness_counts.json` 在完整公平周期统计下各选项计数相等。
4. HPC 模板参数与本机模板一致，仅路径与资源位不同。

产出物：

1. `EdgeFlowNAS/outputs/supernet/train_manifest.json`
2. `EdgeFlowNAS/outputs/supernet/fairness_counts.json`
3. `EdgeFlowNAS/outputs/supernet/eval_epe_history.csv`
4. `EdgeFlowNAS/outputs/supernet/supernet_training_report.md`
5. `EdgeFlowNAS/scripts/hpc/run_supernet_train.sh`

## 7. 失败回滚规则

1. 任一 Gate 失败时，只允许在当前 Gate 内修复，不跨 Gate 堆叠未验证改动。
2. 每次修复后至少回归执行该 Gate 的全部命令一次。
3. Gate-3 失败时禁止直接进入长跑训练，必须先恢复到可重复的 30-step 调试态。

## 8. 当前建议执行起点

1. 先启动 Gate-1。
2. Gate-1 通过后再启动 Gate-2。
3. Gate-2 通过后再进入 Gate-3 的 30-step 调试。
