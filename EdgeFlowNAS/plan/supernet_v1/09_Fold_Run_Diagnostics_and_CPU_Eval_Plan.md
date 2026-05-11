# 09 Fold 训练诊断与修订计划（精简版）
更新时间: 2026-03-03
状态: Superseded（后续执行以 10/11 为准）

## 0. 保留结论
1. CPU 并发/线程评估路线不再作为当前主线。
2. 本轮只做不破坏 checkpoint 恢复能力的最小改动。
3. 主线聚焦为“评估稳定性 + 训练日志可解释性”。

---

## 1. 已选执行项

### P0 固定 eval 口径
执行:
1. Eval 使用固定顺序 batch 采样（固定 index 与 batch 边界）。
2. Eval 使用固定 center crop。
3. BN recal 默认继续使用 train batch，val 仅用于 EPE 评估。

验收:
1. 同一 checkpoint 连续运行 3 次 eval，`mean_epe_12` 波动显著下降。
2. `mean_epe_12/std_epe_12` 跨 epoch 趋势可直接比较。

### P1 训练日志改为 epoch 统计
执行:
1. 在 epoch 内累计 `loss_sum`, `grad_norm_sum`, `step_count`。
2. 在 epoch 结束记录 `loss_epoch_avg`, `grad_norm_epoch_avg`。
3. 快照值仅作为 debug 可选项，不作为主日志。

验收:
1. `loss/grad_norm` 能稳定反映 epoch 级趋势，不再被尾步噪声主导。

### P2 保持项
1. `bn_recal_batches` 机制保持不变，仅保证采样可复现。

---

## 2. 训练口径补充（micro-batch）
1. `micro_batch_size` 不是纯显存参数，会影响 BN 统计与累计梯度轨迹。
2. 当前实现中，梯度裁剪已是“累积后一次 batch 级裁剪”，不是 micro-step 级裁剪。
3. 为保证实验可比性，`batch_size` 与 `micro_batch_size` 需要固定。

---

## 3. 最终验收标准
1. 同一 checkpoint 的重复 eval 方差显著下降。
2. 训练日志的 `loss/grad_norm` 可稳定反映 epoch 趋势。
3. 不影响当前 `run1_folder` 的断点恢复与继续训练。

---

## 4. 归档执行细节（保留 2026-02 长版说明）

### 4.1 当时确认过的范围映射
1. 旧口径问题在 `run1_folder` 恢复训练后不再作为第一优先级。
2. eval 需要尽量固定到确定 batch，优先考虑 `/val` 顺序取前若干 batch。
3. CPU 并发评估不再考虑。
4. `loss/grad_norm` 改为 epoch 均值。
5. `bn_recal_batches` 机制保持不变。
6. 需要基于子网池 `per_arch_epe` 计算 Kendall tau 或等价排名稳定性指标。
7. eval 改用固定 center crop。

### 4.2 当时拟定的执行顺序
1. 先做固定 eval batch。
2. 再做 eval center crop。
3. 再做 epoch 平均训练日志。
4. 再补排名稳定性指标。
5. 保持 `bn_recal_batches` 机制不变，先观察收益。

### 4.3 关于 micro-batch 的旧纠偏结论
1. `micro_batch_size` 不是纯显存参数，也会影响训练动力学。
2. 旧实现下 BN 统计依赖 micro-batch 前向过程，微批大小变化会改变统计量。
3. 当时的梯度裁剪若在 micro-step 粒度执行，也不等价于“大 batch 一次求梯度再裁剪”。
4. 因而要做可比实验，`batch_size` 与 `micro_batch_size` 都需要固定。

### 4.4 深度选项的包含关系（归档结论）
1. `Deep1/Deep2/Deep3` 在实现上是递进包含关系，不是互相独立的三套完全分离分支。
2. 旧代码解释为：
- `out1 = deep1(inputs)`
- `out2 = deep2(out1)`
- `out3 = deep3(out2)`
- 最终按索引在 `[out1, out2, out3]` 中选择
3. 这意味着：
- 选项 `1` 包含选项 `0` 的计算路径
- 选项 `2` 包含选项 `1` 和 `0` 的计算路径
4. 当时结论是不建议在恢复训练链路中改掉这一结构，因为那会破坏 checkpoint 的直接兼容性。
