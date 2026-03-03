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
