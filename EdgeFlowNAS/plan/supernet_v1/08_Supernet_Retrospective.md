# EdgeFlowNAS Supernet 复盘（归档精简版）
更新时间: 2026-03-03
状态: Superseded（执行以 10/11/14 为准）

## 0. 关键结论
1. 当时的主问题不是梯度本身，而是验证数据口径失效（`val` 无效 + synthetic fallback）。
2. 在该口径下，`mean_epe_12/std_epe_12` 与早停判据都会失真。
3. `fairness_gap=0.00` 持续成立，说明公平采样机制本身并非故障源。
4. 处理方向已确定并执行：切换为 folders-only（`train_dir/val_dir`）并移除 synthetic fallback。

---

## 1. 保留证据（摘要）
1. 数据检查显示：`FC2_train` 有效，`FC2_test` 在当时环境下基本无效。
2. 代码链路显示：样本空/读取失败时会走 `_synthetic_triplet(...)`，评估会受到污染。
3. 早停触发在当时配置下符合机制，不属于实现 bug。

---

## 2. 已落地动作
1. 数据读取统一到目录扫描：`data.train_dir` / `data.val_dir`。
2. 删除 synthetic fallback，数据异常改为显式报错。
3. 训练入口加入样本计数校验，`val_samples==0` 时阻断运行。
4. 恢复训练采用新实验名并重置早停状态，避免跨口径直接比较。

---

## 3. 关联文档
1. 当前主线：`10_Eval_Consistency_and_Clip_Diagnostics_Plan.md`
2. 续训决策：`11_200Epoch_结果诊断与续训决策计划.md`
3. 训练细节对齐：`14_单模型与Supernet训练细节对比及数据利用率优化计划.md`
4. 实现记录：`06_Implementation_Log.md`

---

## 4. 归档详证（保留 2026-02 长版证据链）

### 4.1 Data list 有效性检查结果
1. 来自 `check_fc2_data_list_validity.py` 的旧检查结果显示：
- `FC2_dirnames.txt`: `total_entries=22232`, `img0_exists=17785`, `missing_img0=4447`
- `FC2_train.txt`: `img0_exists=17785`
- `FC2_test.txt`: `img0_exists=0`
2. 这说明当时训练环境里的验证 split 实际不可用。
3. `missing_img0=4447` 与 `FC2_test=4446` 只差 1，说明缺失样本集合与旧 test 索引几乎重合。

### 4.2 索引映射结果
1. 旧 `FC2_test.txt` 映射回 `FC2_dirnames.txt` 后，路径前缀是 `Datasets/FlyingChairs2/train/...`，而不是 `val/...` 或 `test/...`。
2. 这更像是“data list 与当前数据落盘版本不匹配”，不只是目录命名问题。

### 4.3 代码层证据
1. 旧配置允许回退：`allow_synthetic_fallback: true`。
2. 数据层在样本空或读取失败时会回退 `_synthetic_triplet(...)`。
3. 评估阶段通过 `val_provider.next_batch(...)` 计算 EPE，因此会被 synthetic 数据污染。

### 4.4 对原判断的纠偏
1. 当时“评估不一致导致指标失效”的方向是对的，但更上游的根因是 `val` 无效导致 synthetic fallback。
2. 日志里的 `grad_norm` 是裁剪前瞬时值而非 epoch 平均值，不能单凭这一项认定“梯度爆炸”是主故障源。
3. BN 重估批数与 eval pool 大小仍是有效优化方向，但都排在“修复真实验证数据来源”之后。

### 4.5 旧日志窗口（epoch 1~49）解读
1. `fairness_gap=0.00` 一直成立，说明公平采样机制本身正常。
2. `mean_epe_12/std_epe_12` 剧烈波动与评估数据异常一致。
3. `epoch=27` 的早停在当时配置 `patience=15, min_delta=0.002` 下可以严格复算，不属于实现 bug。
4. 恢复训练能继续到 `epoch=28~49`，说明恢复链路本身正常；异常来自评估口径而非 resume 逻辑。

### 4.6 当时提出的优先级修复项
P0：
1. 修复 `val` 数据来源。
2. 删除 synthetic fallback。
3. 启动时强制打印 `train_samples/val_samples`，并在 `val_samples==0` 时直接失败。
4. 禁止 supernet 选模直接使用最终测试集。

P1：
1. 固定评估子集或固定评估 seed。
2. 在算力允许时提高 `eval_pool_size` 与 `bn_recal_batches`。

P2：
1. 增加 epoch 级平均训练日志，如 `train_loss_epoch_avg`、`train_grad_norm_epoch_avg`。

P3：
1. 增加 `min_epochs_before_early_stop`，或临时调大 `early_stop_patience`。

### 4.7 与旧 checkpoint 的兼容建议（归档）
1. 旧 checkpoint 权重可恢复，但跨数据口径阶段的 `best_metric/bad_epochs` 不应直接比较。
2. 当时建议新开 `experiment_name` 并通过 `resume_experiment_name` 继承权重，同时重置早停状态。
