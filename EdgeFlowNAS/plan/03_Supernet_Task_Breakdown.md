# 03 超网训练任务拆分与验收标准

更新时间: 2026-02-16

## 1. 目标

将 `02_Supernet_Training_Spec.md` 落成可执行工程任务，聚焦 **Supernet 训练**。  
本文件只定义任务拆分、测试方法和通过标准，不涉及搜索与子网重训实现。

## 2. 前置约束（继承 02，默认已锁定）

1. 训练栈：TF1 兼容链路（沿用 EdgeFlowNet 风格）。
2. 分辨率：`H=180, W=240`。
3. 训练数据：FC2。
4. 验证来源：`FC2_test`（仅用于 supernet 排名阶段）。
5. Fairness：3-path same-batch 梯度累积后单次更新。
6. Optimizer：`Adam + cosine decay`。
7. Weight Decay：`4e-5`。
8. Gradient Clip：global norm `5.0`。
9. BN：共享 BN + 评估前重估，重估 `8 batches`。
10. 验证子网池：`12` 个，分层覆盖采样。
11. 评估频率：每 `1 epoch`。
12. 早停：`patience=15`, `min_delta=0.002`。
13. 预算：`200 epochs`。
14. 随机种子：单种子固定。

## 3. 任务总览

1. T01：工程骨架与配置入口
2. T02：架构编码器（9 维 arch_code）
3. T03：Supernet 网络实现（Bilinear 骨架可切换）
4. T04：Strict Fairness 采样器
5. T05：训练循环（3-path 累积 + 单次更新）
6. T06：验证器（12 子网 + BN 重估 + EPE）
7. T07：早停、checkpoint、断点续训
8. T08：日志与可复现清单（manifest）
9. T09：端到端 smoke 与 HPC 前置验收

## 4. 任务明细与验收测试

## T01 工程骨架与配置入口

目标：

1. 在 `EdgeFlowNAS` 内建立 supernet 专用训练入口。
2. 兼容现有单模型命令风格，新增 supernet 参数。

计划文件：

1. `EdgeFlowNAS/wrappers/run_supernet_train.py`
2. `EdgeFlowNAS/code/train_supernet.py`
3. `EdgeFlowNAS/configs/supernet_fc2_180x240.yaml`

验收测试：

1. 命令：`python wrappers/run_supernet_train.py --help`
- 通过标准：帮助信息包含 `--gpu_device --num_epochs --batch_size --lr --config --supernet_mode`。

2. 命令：`python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --dry_run`
- 通过标准：输出配置解析结果，且不进入训练。

失败判定：

1. 参数缺失导致程序崩溃。
2. 命令参数与配置文件优先级不明确。

## T02 架构编码器（9维）

目标：

1. 定义 `arch_code` 的解析、校验、可读化输出。

计划文件：

1. `EdgeFlowNAS/code/nas/arch_codec.py`

验收测试：

1. 输入 `arch_code=[0,1,2,0,0,1,2,1,0]`
- 通过标准：可解析到 `EB/DB 深度 + H* kernel` 的可读结构。

2. 输入非法长度（如 8 维）
- 通过标准：抛出明确错误，错误信息含“length=9 required”。

3. 输入非法取值（如 3）
- 通过标准：抛出明确错误，错误信息含“value must be in {0,1,2}”。

失败判定：

1. 非法输入静默通过。
2. 解析结果和编码定义不一致。

## T03 Supernet 网络实现

目标：

1. 基于 Bilinear 骨架实现可按 `arch_code` 动态切换的超网前向。

计划文件：

1. `EdgeFlowNAS/code/network/MultiScaleResNet_supernet.py`

验收测试：

1. 命令：`python -m code.nas.model_shape_check --h 180 --w 240 --batch 2`
- 通过标准：随机 20 组 `arch_code` 前向全部成功。

2. 输出尺寸检查
- 通过标准：三尺度输出尺寸分别为 `45x60`, `90x120`, `180x240`（通道按设计）。

3. 参数冻结检查
- 通过标准：除深度与 kernel 选择外，其他结构参数在不同 `arch_code` 下不发生 schema 变化。

失败判定：

1. 任一合法 `arch_code` 前向失败。
2. 输出分辨率与规范不一致。

## T04 Strict Fairness 采样器

目标：

1. 实现 9 个 block 的严格公平采样，周期内每个选项恰好出现一次。

计划文件：

1. `EdgeFlowNAS/code/nas/fair_sampler.py`

验收测试：

1. 命令：`python -m code.nas.fair_sampler --cycles 300 --seed 42 --check`
- 通过标准：每个 block 上 `count(option0)==count(option1)==count(option2)==300`。

2. 可复现检查
- 通过标准：同 seed 下前 100 个周期采样结果完全一致。

失败判定：

1. 出现不公平计数（gap > 0）。
2. 同 seed 结果不一致。

## T05 训练循环（3-path累积 + 单次更新）

目标：

1. 单个公平周期执行 `3` 次前后向、`1` 次 optimizer step。
2. 启用 `Adam + cosine decay + wd=4e-5 + clip=5.0`。

计划文件：

1. `EdgeFlowNAS/code/train_supernet.py`

验收测试：

1. 命令：`python wrappers/run_supernet_train.py --config ... --max_epochs 1 --max_steps 30 --debug_trace`
- 通过标准：日志显示每周期 `forward/backward x3`、`optimizer_step x1`。

2. 梯度裁剪检查
- 通过标准：日志记录 `global_grad_norm_before` 和 `after`，`after <= 5.0`。

3. 学习率调度检查
- 通过标准：lr 随 step 单调按 cosine 变化（允许浮点误差）。

失败判定：

1. 每周期 optimizer step 次数不是 1。
2. 裁剪未生效或调度器未生效。

## T06 验证器（12子网 + BN重估 + EPE）

目标：

1. 每 epoch 对固定 12 子网评估 EPE。
2. 每个子网评估前执行 BN 重估 `8 batches`。

计划文件：

1. `EdgeFlowNAS/code/nas/supernet_eval.py`
2. `EdgeFlowNAS/outputs/supernet/eval_pool_12.json`

验收测试：

1. 验证池覆盖检查
- 通过标准：9 个 block 的每个选项在 12 子网池中均出现至少 1 次，且总分布满足分层覆盖规则。

2. BN 重估执行检查
- 通过标准：每个子网评估日志包含 `bn_recal_batches=8`，并记录重估前后 BN 统计摘要。

3. 指标产出检查
- 通过标准：每个 epoch 都写入 `mean_epe_12`、`std_epe_12`、`fairness_gap`。

失败判定：

1. 验证池未固定或覆盖不达标。
2. BN 重估被跳过。

## T07 早停、checkpoint、断点续训

目标：

1. 按 `patience=15, min_delta=0.002` 执行早停。
2. 保留 `best/last` checkpoint 并支持续训。

计划文件：

1. `EdgeFlowNAS/code/train_supernet.py`
2. `EdgeFlowNAS/outputs/supernet/checkpoints/`

验收测试：

1. 早停逻辑单元测试（注入模拟 metric 序列）
- 通过标准：触发 epoch 与理论值一致。

2. 断点续训测试
- 步骤：训练 3 epochs -> 中断 -> resume 到 5 epochs。
- 通过标准：global step 连续，optimizer/scheduler 状态连续。

失败判定：

1. resume 后重复或跳过 step。
2. best checkpoint 指向错误。

## T08 日志与可复现清单

目标：

1. 保存完整运行元信息，支持实验追踪。

计划文件：

1. `EdgeFlowNAS/outputs/supernet/train_manifest.json`
2. `EdgeFlowNAS/outputs/supernet/fairness_counts.json`
3. `EdgeFlowNAS/outputs/supernet/eval_epe_history.csv`

验收测试：

1. manifest 字段检查
- 通过标准：包含 `seed, config_hash, git_commit, input_shape, optimizer, lr_schedule, wd, grad_clip`。

2. fairness 计数检查
- 通过标准：训练结束时每个 block 三选项计数相等（按完整周期统计）。

失败判定：

1. 关键字段缺失。
2. fairness 统计文件缺失或不一致。

## T09 端到端 Smoke 与 HPC 前置验收

目标：

1. 在本机 `tf_work_hpc` 跑通最小可用链路。
2. 形成 HPC 可执行命令模板。

验收测试：

1. 本机 smoke
- 命令：`python wrappers/run_supernet_train.py --gpu_device 0 --num_epochs 2 --batch_size 32 --lr 1e-4 --config configs/supernet_fc2_180x240.yaml --fast_mode`
- 通过标准：完成 2 epochs，产出 checkpoint 与 eval 日志。

2. HPC 命令模板检查
- 通过标准：模板参数与本机命令一致，仅替换数据根路径、日志路径、资源参数。

失败判定：

1. 本机无法完整跑通 2 epochs。
2. HPC 模板缺失关键参数。

## 5. 里程碑与完成定义

M1（基础可跑）：

1. T01-T04 全通过。

M2（训练可用）：

1. T05-T07 全通过。

M3（交付就绪）：

1. T08-T09 全通过。
2. 输出完整 artifact：
3. `supernet_best.ckpt`
4. `supernet_last.ckpt`
5. `train_manifest.json`
6. `fairness_counts.json`
7. `eval_epe_history.csv`

## 6. 风险与缓解

1. 风险：三路径累积导致显存峰值波动。
- 缓解：保持单路径串行前后向；必要时减小 batch 并放宽日志频率。

2. 风险：共享 BN 导致子网评估噪声。
- 缓解：固定 BN 重估批次为 8，并记录重估前后差异。

3. 风险：`FC2_test` 用作 supernet 验证引入“测试集参与选择”争议。
- 缓解：文档显式限定其用途仅为 supernet 阶段相对排序；最终子网结论在后续独立训练阶段给出。
