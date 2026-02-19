# 10 Eval 一致性与 Clip 诊断计划
更新时间: 2026-02-17

## 0. 目标与范围
1. 本文档仅关注超网训练过程稳定性与评估口径一致性。  
2. 子网搜索/最终子网选择暂不纳入本轮讨论。  

---

## 1. 已完成改动（当前基线）

### 1.1 Eval 一致性（已完成）
1. Eval 数据改为固定顺序采样。  
2. Eval 裁剪改为固定中心裁剪。  
3. 训练数据仍保持随机采样与随机裁剪。  

涉及文件:
- `MCUFlowNet/EdgeFlowNAS/code/data/fc2_dataset.py`
- `MCUFlowNet/EdgeFlowNAS/code/data/dataloader_builder.py`

### 1.2 Eval 覆盖增强（已完成）
1. 新增 `eval_batches_per_arch`。  
2. 当前默认值 `4`。  
3. 每个 arch 的 EPE 由 4 个固定 val batch 求均值。  

涉及文件:
- `MCUFlowNet/EdgeFlowNAS/configs/supernet_fc2_180x240.yaml`
- `MCUFlowNet/EdgeFlowNAS/code/engine/supernet_trainer.py`
- `MCUFlowNet/EdgeFlowNAS/code/nas/supernet_eval.py`
- `MCUFlowNet/EdgeFlowNAS/wrappers/run_supernet_eval.py`

### 1.3 BN 重估一致性（已完成）
1. Eval 模式下 BN recal 使用固定顺序 train batch。  
2. 每个 arch 在 BN recal 前与 val eval 前都会 reset 游标。  

涉及文件:
- `MCUFlowNet/EdgeFlowNAS/code/engine/supernet_trainer.py`
- `MCUFlowNet/EdgeFlowNAS/code/nas/supernet_eval.py`

### 1.4 日志增强（已完成）
1. 新增梯度分位数日志：`train_grad_norm_p50/p90/p99`。  
2. 新增每轮 12 子网排名日志：`arch_rank_12`（`rank:arch_index:epe`）。  
3. 未加入 `clip_checks` / `expected_clip_checks`（按当前需求）。  
4. 未加入 `kendall_tau_prev`（按当前需求，改为直接输出排名）。  

涉及文件:
- `MCUFlowNet/EdgeFlowNAS/code/engine/supernet_trainer.py`

### 1.5 Clip 机制改动（已完成）
1. 梯度裁剪由“micro-step 级”改为“累积后一次 batch 级裁剪”。  
2. `grad_clip_global_norm` 已设置为 `200.0`。  

涉及文件:
- `MCUFlowNet/EdgeFlowNAS/code/engine/supernet_trainer.py`
- `MCUFlowNet/EdgeFlowNAS/configs/supernet_fc2_180x240.yaml`

---

## 2. 新日志分析（Epoch 104-109）

日志摘要:
1. `clip_rate` 从过去的 1.0000 下降到 `0.0000 ~ 0.0800`。  
2. `clip_count` 在每 epoch 仅 `0~4`（对应当前 step 级统计）。  
3. `grad_norm` 大约 `60~102`，`grad_p90` 大约 `104~189`。  
4. `mean_epe_12` 位于 `9.758 ~ 9.824`，整体较 95-97 轮偏高。  
5. `std_epe_12` 在 104/105/107 轮较低（约 0.02~0.04），在 106/108/109 轮明显升高（约 0.075~0.163）。  
6. `fairness_gap` 持续为 `0.00`，公平采样机制仍正常。  

### 2.1 可确认的正向结论
1. “裁剪全饱和”问题已解除。  
2. 训练更新不再长期被硬阈值压死。  
3. Batch 级一次裁剪策略在实现层面与预期一致。  

### 2.2 当前主要风险
1. 虽然 clip 退饱和，但 `mean_epe_12` 没有出现稳定下降。  
2. `std_epe_12` 间歇性放大，说明 12 子网中尾部子网波动/退化较明显。  
3. 从 `arch_rank_12` 看，`arch_index=9` 连续处于最差且多次恶化到 `10+`，是方差放大的主要来源之一。  
4. `arch_index=4` 也常驻后段，属于次级风险子网。  

### 2.3 对当前阶段的解释
1. 之前的瓶颈（过强裁剪）已被移除。  
2. 现在的瓶颈更像“优化动力学偏激进”而非“梯度被截断”。  
3. 由于 batch 级裁剪 + 高阈值（200）显著放松约束，当前学习率 `1e-4` 可能偏高，导致尾部子网漂移。  

---

## 3. 下一步建议（仅超网训练，不涉及子网选择）

## P0（推荐先做）：在保持 batch 级裁剪不变的前提下收敛优化动力学

### P0-A 学习率下调（优先级最高）
目标:
在不回退 clip 机制的前提下，抑制尾部子网漂移。

建议:
1. 将 `lr` 从 `1e-4` 下调到 `5e-5` 先跑 8-12 个 epoch。  
2. 若仍有明显尾部抖动，再下调到 `3e-5`。  

判据:
1. `mean_epe_12` 恢复下降或至少回到 104 附近。  
2. `std_epe_12` 长期回落到 <= `0.04` 区间。  
3. `arch_index=9` 的 EPE 不再频繁冲到 `10+`。  

### P0-B Clip 阈值微收紧（第二优先级）
目标:
在“不过饱和”前提下增加一点约束，降低极端步长。

建议:
1. 保留 batch 级裁剪。  
2. 将阈值由 `200` 试到 `160`，必要时试 `120`。  

判据:
1. `clip_rate` 保持低但非零（例如 `0.02~0.15` 区间可接受）。  
2. `mean_epe_12` 不劣化，`std_epe_12` 收敛更稳。  

---

## P1（若 P0 无效再做）：加监控，不先改结构

### P1-A 低成本日志补充（可选）
建议新增:
1. `worst_arch_epe` 与 `worst_arch_idx`（直接定位尾部波动源）。  
2. `best_arch_epe` 与 `best_arch_idx`（观察头部是否同步改善）。  
3. `tail_gap = worst_arch_epe - median_arch_epe`（衡量排名尾部拖累程度）。  

说明:
这些是统计层补充，不影响训练图效率。  

---

## 4. 实验执行协议（下一轮）

1. 每次只改一个主变量（先改 LR，再改 clip 阈值）。  
2. 保持以下条件固定:
- `batch_size`
- `micro_batch_size`
- `eval_batches_per_arch=4`
- `bn_recal_batches`
- 数据路径与 eval 配置
3. 每档至少跑 8-12 epoch 再判定。  
4. 对比指标最少包括:
- `mean_epe_12`
- `std_epe_12`
- `clip_rate`
- `grad_norm` / `grad_p90`
- `arch_rank_12` 尾部稳定性

---

## 5. 兼容性说明
1. 当前改动不改变模型变量形状，checkpoint 兼容。  
2. 可继续断点恢复训练。  
3. 做趋势对比时需标注“裁剪机制切换点”和“阈值切换点”。  
