# 10 Eval 一致性与 Clip 诊断计划（精简版）
更新时间: 2026-03-03
状态: Active

## 0. 目标与范围
1. 仅关注超网训练稳定性与 eval 口径一致性。
2. 不讨论最终子网选择。

---

## 1. 已完成基线

### 1.1 Eval 一致性
1. Eval 数据改为固定顺序采样。
2. Eval 裁剪改为固定 center crop。
3. Train 保持随机采样与随机裁剪。

涉及文件:
- `MCUFlowNet/EdgeFlowNAS/code/data/fc2_dataset.py`
- `MCUFlowNet/EdgeFlowNAS/code/data/dataloader_builder.py`

### 1.2 Eval 覆盖
1. 已加入 `eval_batches_per_arch`，默认值为 `4`。
2. 每个 arch 的 EPE 由 4 个固定 val batch 求均值。

涉及文件:
- `MCUFlowNet/EdgeFlowNAS/configs/supernet_fc2_180x240.yaml`
- `MCUFlowNet/EdgeFlowNAS/code/engine/supernet_trainer.py`
- `MCUFlowNet/EdgeFlowNAS/code/nas/supernet_eval.py`
- `MCUFlowNet/EdgeFlowNAS/wrappers/run_supernet_eval.py`

### 1.3 BN recal 一致性
1. Eval 下 BN recal 使用固定顺序 train batch。
2. 每个 arch 在 BN recal 前与 val eval 前均 reset 游标。

涉及文件:
- `MCUFlowNet/EdgeFlowNAS/code/engine/supernet_trainer.py`
- `MCUFlowNet/EdgeFlowNAS/code/nas/supernet_eval.py`

### 1.4 Clip 与日志
1. 梯度裁剪已改为“累积后一次 batch 级裁剪”。
2. `grad_clip_global_norm=200.0`。
3. 日志包含 `train_grad_norm_p50/p90/p99` 与 `arch_rank_12`。

涉及文件:
- `MCUFlowNet/EdgeFlowNAS/code/engine/supernet_trainer.py`
- `MCUFlowNet/EdgeFlowNAS/configs/supernet_fc2_180x240.yaml`

---

## 2. 诊断结论（Epoch 104-109）
1. `clip_rate` 已从历史 1.0000 降到 `0.0000~0.0800`，饱和问题解除。
2. `mean_epe_12` 未稳定下降，仍有波动（`9.758~9.824`）。
3. `std_epe_12` 存在间歇性放大（最高约 `0.163`），尾部子网不稳。
4. 主要风险子网集中在 `arch_index=9`（次级风险为 `arch_index=4`）。

---

## 3. 已选下一步（唯一执行路径）
### P0 学习率下调
执行:
1. `lr: 1e-4 -> 5e-5`，先跑 8-12 epoch。
2. 若尾部仍明显抖动，再调到 `3e-5`。

判据:
1. `mean_epe_12` 恢复下降或至少回到 104 附近。
2. `std_epe_12` 回落并稳定在 `<=0.04` 区间。
3. `arch_index=9` 不再频繁恶化到 `10+`。

---

## 4. 实验执行协议
1. 每次只改一个主变量（本轮仅改 LR）。
2. 固定条件: `batch_size`、`micro_batch_size`、`eval_batches_per_arch=4`、`bn_recal_batches`、数据路径与 eval 配置。
3. 每档至少跑 8-12 epoch 再判定。
4. 对比指标: `mean_epe_12`、`std_epe_12`、`clip_rate`、`grad_norm/grad_p90`、`arch_rank_12` 尾部稳定性。

---

## 5. 兼容性
1. 当前改动不改变模型变量形状，checkpoint 兼容。
2. 可继续断点恢复训练。
3. 趋势对比需标注“裁剪机制切换点”。
