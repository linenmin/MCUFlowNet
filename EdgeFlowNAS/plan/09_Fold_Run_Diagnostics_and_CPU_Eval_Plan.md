# 09 Fold 训练诊断与修订计划（更新版）
更新时间: 2026-02-17

## 0. 本版结论
1. 放弃 CPU 并发/线程评估路线，不再作为当前阶段主线。
2. 主线聚焦为“评估稳定性 + 训练日志可解释性 + 子网排序稳定性”。
3. 在不破坏现有 checkpoint 恢复能力的前提下做最小改动。

---

## 1. 用户确认后的范围（逐条映射）
1. `P0-1`：你已使用 `edgeflownas_supernet_fc2_180x240_run1_folder` 恢复训练，旧口径问题当前不再优先。
2. `P0-2`：评估数据尽量固定到确定 batch（你建议从 `/val` 顺序取前若干 batch）。
3. `P0-3`：不再考虑 CPU 并发评估方案。
4. `P1-5`：训练日志中的 `loss` / `grad_norm` 改为 epoch 均值（替换快照值）。
5. `P1-6`：`bn_recal_batches` 机制保持不变（先不做结构性改动）。
6. `P2-8`：基于子网池 `per_arch_epe` 计算 Kendall tau，并写入日志。
7. `P2-9`：评估改用固定中心裁剪（降低随机性）。
8. `P3-10`：对 micro-batch 影响进行澄清（见第 4 节）。

---

## 2. 现在要做的计划（按优先级）

## P0 评估稳定化（必须先做）
### P0-A 固定评估 batch（Deterministic Eval Batches）
目标:
1. 同一 checkpoint 多次评估结果波动显著下降。
2. `mean_epe_12/std_epe_12` 更可比较，减少“随机 batch 噪声”。

执行逻辑:
1. 为 eval 增加“固定批次采样器”（按文件顺序、固定 index、固定 batch 边界）。
2. 评估阶段不再每次随机 `next_batch`。
3. `eval_pool` 仍保持固定（已有）。

说明:
1. 你提出用 `/val` 前 `bn_recal_batches` 个 batch。可以实现。
2. 但建议默认仍使用 `train` 做 BN recal，`val` 只做 EPE，避免“在验证集上做 BN 统计适配”导致指标偏乐观。

验收:
1. 同一 checkpoint 连续运行 3 次 eval，`mean_epe_12` 相对波动显著下降。

### P0-B 评估裁剪固定为中心裁剪
目标:
1. 进一步压低随机裁剪噪声。

执行逻辑:
1. 在 eval 路径中使用 center crop。
2. train 路径保持随机裁剪，不改训练增强。

验收:
1. eval 方差下降，且指标趋势更平滑。

## P1 训练日志可解释性
### P1-A `loss` / `grad_norm` 记录 epoch 均值
问题:
1. 当前日志是最后一次 micro-step 快照，解释性弱。

执行逻辑:
1. 在 epoch 内累计 `loss_sum`, `grad_norm_sum`, `step_count`。
2. epoch 结束时记录 `loss_epoch_avg`, `grad_norm_epoch_avg`。
3. 快照值可作为 debug 可选项保留，但主日志用均值。

验收:
1. 日志字段可直接用于跨 epoch 对比，不再受尾步噪声主导。

## P1 子网排序稳定性指标
### P1-B 增加 Kendall tau（基于 `per_arch_epe`）
目标:
1. 直接衡量“子网排名是否稳定”，补齐仅看均值的盲区。

执行逻辑:
1. 每轮 eval 后基于 `per_arch_epe` 得到当前排名。
2. 与上一轮排名计算 `tau_prev`（必要）。
3. 可选：与“历史 best 轮排名”计算 `tau_best`（增强可解释性）。

验收:
1. 日志中新增 `kendall_tau_prev`（至少一个 tau 指标）。

## P1 保持项
### P1-C `bn_recal_batches` 保持原样
说明:
1. 本轮不改 BN recal 机制，只改“采样是否固定/可复现”。

---

## 3. 训练本身（非 eval）建议检查项
以下是除 eval 外，建议优先检查的训练项：

1. 训练统计口径：
- 主日志换成 epoch 平均（第 2 节已覆盖）。

2. `micro_batch_size` 影响（见第 4 节）：
- 不建议把它当“纯内存参数”看待，建议固定一个值（例如 8）保持实验可比。

3. 配置一致性：
- `eval_every_epoch` 配置建议实装或删除其无效字段，避免“配置写了但不生效”。

4. 可复现性增强（可选）：
- 若后续要严格复现实验，需考虑保存/恢复数据采样器 RNG 状态（当前 resume 不恢复数据游标状态）。

---

## 4. 关于 micro-batch 的结论（纠偏）
结论：`micro_batch_size` 不只是影响显存，它会影响训练动力学，因此可能影响 loss/EPE。

依据（当前实现）:
1. BN 更新发生在每个 micro-batch 前向中，统计量受 micro-batch 大小影响。
2. 梯度裁剪在 micro-batch 粒度执行后再累计，不等价于“大 batch 一次性求梯度再裁剪”。

所以:
1. 仅改变 `micro_batch_size`，即使 `batch_size` 不变，也可能改变收敛轨迹。
2. 要做可比实验，建议固定 `micro_batch_size`。

---

## 5. 深度选项 0/1/2 是否包含关系（代码结论）
结论：是包含关系，而且是严格递进包含。

代码证据:
1. `EdgeFlowNAS/code/network/MultiScaleResNet_supernet.py` 的 `_deep_choice_block` 中：
- `out1 = deep1(inputs)`
- `out2 = deep2(out1)`
- `out3 = deep3(out2)`
- 最后在 `[out1, out2, out3]` 中按索引选择。

含义:
1. 选项 `1` 包含了选项 `0` 的计算路径。
2. 选项 `2` 包含了选项 `1` 和 `0` 的计算路径。

优点:
1. 参数共享强，搜索训练更稳定。
2. “深度增加”语义单调，易解释。
3. 与 strict fairness 配合简单。

缺点:
1. 选项独立性弱，排名可能受共享路径耦合影响。
2. 超网训练时三条候选都构图并计算，训练期真实算力差异不明显。
3. 可能带来 supernet ranking 与离散子网独训 ranking 偏差。

是否需要改:
1. 当前阶段不建议改。
2. 原因：改成“非包含独立分支”会改变参数拓扑，基本无法无缝继承当前 checkpoint，需重训并重新建立对比基线。
3. 建议作为下一阶段（新实验名）单独研究项，而不是插入当前恢复训练链路。

---

## 6. 实施顺序（最终）
1. 先做固定 eval batch + center crop（P0）。
2. 再做训练日志 epoch 平均化（P1-A）。
3. 加 Kendall tau 日志（P1-B）。
4. 保持 `bn_recal_batches` 机制不变，先观察一轮稳定性收益。
5. 深度选项结构保持现状，不在本轮动架构。

---

## 7. 验收标准
1. 同一 checkpoint 重复评估结果方差显著下降。
2. 训练日志中的 `loss/grad_norm` 可稳定反映 epoch 趋势。
3. 每轮日志可看到 Kendall tau（至少 `tau_prev`）。
4. 不影响当前 `run1_folder` 的断点恢复与继续训练。
