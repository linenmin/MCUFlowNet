# 10 Eval 一致性与 Clip 诊断计划
更新时间: 2026-02-17

## 0. 本轮确认结论
1. 当前代码里，eval 还不是“固定 batch + 中心裁剪”。
2. `clip_rate=1` 在你这段日志里是“数值上正常、训练上可疑”的信号：说明几乎每次都触发了梯度裁剪。
3. 目前日志中的 `mean_epe_12` 不稳步下降，核心不是单一问题，而是“评估噪声 + 裁剪强度 + 指标覆盖不足”叠加。

代码定位:
- `MCUFlowNet/EdgeFlowNAS/code/engine/supernet_trainer.py` 的 `_run_eval_epoch()` 仍调用 `next_batch()`。
- `MCUFlowNet/EdgeFlowNAS/code/nas/supernet_eval.py` 也仍调用 `next_batch()`。
- `MCUFlowNet/EdgeFlowNAS/code/data/fc2_dataset.py` 里 `_load_one()` 是随机采样，`_random_crop_triplet()` 是随机裁剪。

---

## 1. P0-1: Eval 可重复性不足（最高优先级）

问题具体描述:
当前 eval 每个 arch 都在随机抽样、随机裁剪，导致同一 checkpoint 的评估值有较大采样噪声，epoch 间趋势可解释性弱。

当时实现考虑:
统一 `next_batch()` 路径实现简单，训练与评估代码复用高，且早期开发阶段更关注“先跑通”。

现在的缺点:
`mean_epe_12/std_epe_12` 变化混入了较大的数据采样噪声，难以判断模型本身是否改善。

优化选项 A（推荐）:
给 provider 增加 eval 模式，按固定顺序取样 + 中心裁剪；训练仍保留随机采样 + 随机裁剪。
优点: 改动小、可复现性明显提升、与现有 checkpoint 兼容。
缺点: 需要在 `fc2_dataset.py` 增加模式参数与顺序游标。

优化选项 B:
离线固化 eval batch（保存样本路径和裁剪窗口）到 JSON/NPY，训练和独立 eval 都读取同一批次。
优点: 可复现性最强，跨机器一致。
缺点: 增加一份“评估工件”管理成本。

优化选项 C:
每次 eval 直接跑完整 val（固定中心裁剪）。
优点: 指标方差最低，最接近真实泛化表现。
缺点: 评估开销显著上升。

---

## 2. P0-2: `clip_rate=1` 持续满触发

问题具体描述:
你的日志中连续多个 epoch 都是 `clip_rate=1.0000`，说明每次检查都触发裁剪。当前配置 `grad_clip_global_norm=5.0`，而日志 `grad_norm` 常在 90~140，远高于阈值。

当时实现考虑:
早期为了稳健，采用较保守的全局梯度裁剪，防止 strict-fairness 多路径训练时梯度爆炸。

现在的缺点:
如果长期 100% 触发，更新幅度会长期受硬阈值限制，训练可能进入“方向在变、步长总被压扁”的状态，影响收敛速度与排名区分度。

优化选项 A（推荐）:
先做“诊断不改行为”：记录每 epoch 的梯度分布分位数（p50/p90/p99）并保留当前阈值，再决定阈值是否上调。
优点: 风险最低，先证据后改动。
缺点: 需要多跑几轮才能定阈值。

优化选项 B:
将 `grad_clip_global_norm` 从 5 提高到 20/50（逐步试验）。
优点: 快速验证“过强裁剪是否主因”。
缺点: 若直接提太高，可能出现训练不稳。

优化选项 C:
将裁剪位置从“每个 micro-step”改为“累积后一次裁剪”。
优点: 更贴近“大 batch 梯度再裁剪”的语义。
缺点: 改动训练动力学更大，需重新对比基线。

关键校验公式:
`expected_clip_checks = steps_per_epoch * 3 * ceil(batch_size / micro_batch_size)`。
你给出的 `clip_count=150` 若 `steps_per_epoch=50`，则推导 `ceil(batch_size/micro_batch_size)=1`，即有效 micro-batch 看起来像是等于 batch-size。需要核对该次运行开头日志里的 `micro_batch_size`。

---

## 3. P1-3: Eval 覆盖度与稳定性不足

问题具体描述:
当前每个 arch 只用 1 个 val batch 做 EPE，样本覆盖有限，对 batch 组成敏感。

当时实现考虑:
控制 epoch 时长，避免评估拖慢训练。

现在的缺点:
即使修复随机性，单 batch 评估仍可能方差偏大。

优化选项 A（推荐）:
每个 arch 固定评估 `K` 个 val batch（例如 K=2 或 4）再取均值。
优点: 指标更稳；可线性调节耗时。
缺点: eval 时间增加。

优化选项 B:
固定 1 个 batch 但固定样本 ID。
优点: 成本最低。
缺点: 代表性不足，易过拟合该批次。

---

## 4. P1-4: BN Recal 仍是随机批次

问题具体描述:
BN recal 每个 arch 都使用随机 train batch，导致同一 arch 在不同 epoch 的 BN 统计本身也在抖动。

当时实现考虑:
通过随机 train batch 近似真实训练分布，代码简单。

现在的缺点:
它会把额外方差注入 eval 指标，弱化 epoch 间可比性。

优化选项 A（推荐）:
BN recal 改为固定 train 批次集合（按顺序前 `bn_recal_batches` 批）。
优点: 与“固定 val 批次”配套，整体评估更稳。
缺点: 固定批次可能对数据分布覆盖不足。

优化选项 B:
保留随机 BN recal，但记录每次 BN recal 所用样本索引摘要（hash）。
优点: 开销很小，后续可追溯。
缺点: 不能直接降低方差，只能辅助诊断。

---

## 5. P1-5: 建议增加的低开销日志（不明显影响训练效率）

建议新增字段:
1. `clip_checks` 与 `expected_clip_checks`（用于立刻发现 micro-batch 配置不一致）。
2. `grad_norm_p50/p90/p99`（每 epoch 统计一次）。
3. `eval_seconds`、`bn_recal_seconds`（拆分评估耗时瓶颈）。
4. `per_arch_epe_min/median/max`（比只看均值更有诊断力）。
5. `kendall_tau_prev`（基于 12 个子网 `per_arch_epe` 与上一个 eval 的排序一致性）。

这些统计都可在 epoch 级聚合，几乎不增加训练图计算负担。

---

## 6. P2-6: 是否存在训练“硬故障”

当前判断:
1. 从你贴的日志看，没有 NaN/崩溃/发散到不可训练的硬故障迹象。
2. 主要问题是“评估信号噪声过高 + 裁剪长时间满触发”，这会让曲线难看、收敛慢、排名不稳。

---

## 7. 执行顺序（建议）

1. 先做 P0-1: 固定 eval batch + 中心裁剪（仅 eval 路径）。
2. 做 P0-2 的诊断版日志增强，不立即改裁剪阈值。
3. 跑 5~10 个 epoch，观察 `clip_rate`、`grad_norm` 分位数、`kendall_tau_prev`。
4. 若 `clip_rate` 仍接近 1，再做小步阈值试验（5 -> 20 -> 50）。
5. 最后再决定是否实施“累积后一次裁剪”的结构性改动。

---

## 8. 与断点恢复兼容性

兼容且推荐先做:
1. eval 固定批次与中心裁剪（只影响评估输入，不改权重形状）。
2. 新增日志字段（只增观测，不改训练图参数）。

需要谨慎分实验名:
1. 改裁剪阈值、改裁剪位置（会改变训练动力学；checkpoint 可读，但曲线不可与旧 run 直接同口径比较）。
