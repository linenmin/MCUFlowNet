# EdgeFlowNAS Supernet 复盘（纠偏版）

更新时间: 2026-02-17

## 0. 结论摘要

1. 第一核心问题已经确认：`val` 数据在当前 HPC 运行环境中无效，评估阶段触发了 `synthetic fallback`。
2. 因为评估输入不是实际 FC2 验证样本，`mean_epe_12/std_epe_12` 不能作为真实子网排序依据，早停依据也随之失真。
3. 目前日志里的 `loss` 与 `grad_norm` 不是 epoch 平均值，而是每个 step 最后一个 `arch + micro-batch` 的瞬时值，不能直接用于“训练稳定性”定性。
4. `fairness_gap=0.00` 持续成立，说明 Strict Fairness 采样机制本身正常。
5. 最终工程决策：切换为 **folders-only**（直接扫 `train_dir/val_dir`），并移除 `synthetic fallback`，避免静默容错掩盖数据问题。

---

## 1. 证据链（本次问题的直接证据）

### 1.1 Data list 有效性检查结果

来自 `check_fc2_data_list_validity.py`：

- `FC2_dirnames.txt`: `total_entries=22232`, `img0_exists=17785`, `missing_img0=4447`
- `FC2_train.txt`: `img0_exists=17785`（全部有效）
- `FC2_test.txt`: `img0_exists=0`（全部无效）

这说明当前训练环境里，验证 split 实际不可用。
同时，`missing_img0=4447` 与 `FC2_test=4446` 只差 1，说明“缺失样本集合”与 test 索引集合几乎完全重合。

### 1.2 索引映射结果

将 `FC2_test.txt` 映射回 `FC2_dirnames.txt` 后，路径前缀都是 `Datasets/FlyingChairs2/train/...`，而不是 `val/...` 或 `test/...`。在当前落盘目录结构下，这批索引对应文件均不存在。

这更像是“data list 与当前数据落盘版本不匹配”问题，而不只是“目录名叫 train/val/test 的表面问题”。

### 1.3 代码层证据

- 配置允许回退：`EdgeFlowNAS/configs/supernet_fc2_180x240.yaml` 中 `allow_synthetic_fallback: true`
- 数据回退逻辑：`EdgeFlowNAS/code/data/fc2_dataset.py` 的 `_load_one()` 在样本空/读取失败时返回 `_synthetic_triplet(...)`
- 评估读取来源：`EdgeFlowNAS/code/engine/supernet_trainer.py` 的 `_run_eval_epoch()` 使用 `val_provider.next_batch(...)` 来计算 `EPE`

因此当前运行中，评估阶段极可能大量使用 synthetic 数据。

---

## 2. 对原复盘结论的纠偏

### 2.1 “评估不一致导致指标失效”

- 原判断方向是对的，但**根因优先级应调整**：
  - 原文强调的是“随机 batch 方差过大”；
  - 本次证据表明更上游的根因是“`val` 无效导致 synthetic fallback”。
- 结论：应把“`val` 数据有效性”上升为 P0（第一优先级）。

### 2.2 “梯度爆炸”判断需谨慎

- 日志中的 `grad_norm` 是裁剪前 global norm 且为瞬时值（不是 epoch 平均）。
- 可以作为风险提示，但不能单凭当前日志判定为主要故障根因。
- 在 `val` 修复前，不建议把主要精力放在梯度策略调参上。

### 2.3 BN 重估批数 / eval_pool 容量

- 这两点仍然是有效优化方向；
- 但前提是先修复真实验证数据来源，否则调这些会被 synthetic 评估噪声掩盖。

---

## 3. 当前日志（epoch 1~49）解读

1. `fairness_gap=0.00` 一直成立：采样公平性正常。
2. `mean_epe_12/std_epe_12` 剧烈波动（如 std 多次超过 mean）与“评估数据异常”一致。
3. `epoch=27` 触发早停可被严格复算：
   - 当时配置是 `patience=15`, `min_delta=0.002`。
   - 历史最优在 `epoch=12`，`best=0.625291`，后续必须 `< 0.623291` 才算改进。
   - `epoch=13~27` 连续 15 轮都未达到该阈值，因此在 `epoch=27` 触发早停是符合机制的。
4. 你后续恢复到 `epoch=28~49` 且未再次早停，说明“恢复训练链路正常”；若你已把 `early_stop_patience` 提高到更大值（如 200），这一现象完全符合预期。

---

## 4. 更优修复方案（按优先级）

### P0（必须先做）

1. 修复 `val` 数据来源（保证评估使用真实 FC2 验证样本）。
2. 删除 synthetic fallback（训练与评估都不再允许 synthetic 样本兜底）。
3. 训练启动时强制打印并校验：
   - `train_samples` 数量
   - `val_samples` 数量
   - 若 `val_samples==0`，直接报错退出（不要继续训练）。
4. 明确禁止 supernet 选模直接使用最终测试集：优先 `FC2_val` 或 `FC2_train` 内固定 holdout 作为 ranking/eval split。

### P1（评估稳定性）

1. 固定评估子集（fixed eval batches）或固定评估 seed。
2. 提高 `eval_pool_size` 与 `bn_recal_batches`（在算力允许范围内）。

### P2（监控口径）

1. 增加 epoch 级平均训练日志：
   - `train_loss_epoch_avg`
   - `train_grad_norm_epoch_avg`
2. 保留瞬时值作为 debug 信息，但不作为主判断依据。

### P3（早停策略）

1. 增加 `min_epochs_before_early_stop`。
2. 或在修复前临时大幅提高 `early_stop_patience`，避免过早终止。

---

## 5. 数据加载最终方案

最终采用单一方案：

1. 只使用目录扫描（folders-only）：`data.train_dir` / `data.val_dir`。
2. 不再依赖 `FC2_dirnames.txt + FC2_train.txt + FC2_test.txt` 索引链。
3. 启动时输出 train/val 有效样本数，`val_samples==0` 直接失败。
4. 删除 synthetic fallback，任何数据问题都显式报错。

---

## 6. 与“已有超网断点恢复”的兼容建议

1. 旧 checkpoint 权重仍可恢复（参数形状不变）。
2. 但数据来源口径已变化，`best_metric/bad_epochs` 跨阶段不可直接比较。
3. 推荐恢复策略：
   - 新建 `experiment_name`；
   - 使用 `resume_experiment_name=<旧run>` 继承权重；
   - 并设置 `checkpoint.reset_early_stop_on_resume=true`，重新统计早停状态。

---

## 7. 下一步执行清单

1. 先修复验证数据路径/切分（P0）。
2. 训练入口增加 train/val 计数打印与 `val_samples==0` 强校验（P0）。
3. 再进行评估稳定性与早停策略优化（P1/P3）。
4. 最后再评估是否需要调梯度策略（P2 之后再看）。
