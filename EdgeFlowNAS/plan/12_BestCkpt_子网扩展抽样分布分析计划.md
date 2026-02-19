# 12 BestCkpt 子网扩展抽样分布分析计划
更新时间: 2026-02-19

## 0. 目标
1. 明确 `supernet_best.ckpt` 的保存判据。  
2. 在最佳权重下，抽样更多子网并评估 EPE 分布。  
3. 输出可复用的统计结果与图表，支持后续横向对比。  

---

## 1. `supernet_best.ckpt` 保存判据
1. 判据指标是每个 eval epoch 的 `mean_epe_12`（越小越好）。  
2. 仅当 `mean_epe_12 < (best_metric - early_stop_min_delta)` 时，才覆盖保存 `supernet_best.ckpt`。  
3. 这意味着“更小但小于阈值幅度”的改进不会更新 best。  
4. 若恢复训练时使用 `--reset_early_stop_on_resume`，`best_metric` 会重置为 `inf`，之后的 best 记录将以“恢复后的阶段”为基准。  

代码来源:
- `MCUFlowNet/EdgeFlowNAS/code/engine/early_stop.py`
- `MCUFlowNet/EdgeFlowNAS/code/engine/supernet_trainer.py`

---

## 2. 脚本范围与输出

### 2.1 输入
1. supernet 配置文件（默认 `configs/supernet_fc2_180x240.yaml`）。  
2. checkpoint 类型（默认 `best`，可选 `last`）。  
3. 抽样规模、随机种子、BN recal 与 eval batch 设置。  

### 2.2 处理
1. 在 `3^9` 架构空间中做唯一随机抽样（必要时支持全空间穷举）。  
2. 可选将固定 `eval_pool_12` 并入样本，保证与历史口径可对齐。  
3. 逐子网执行 `BN recal + EPE`，收集分布。  
4. 计算复杂度代理分数（统一方向）用于可解释性分析。  

### 2.3 输出
1. `subnet_samples.csv`：每个子网的 `arch_code / epe / complexity_score`。  
2. `summary.json`：均值、方差、分位数、top/bottom-k。  
3. 图表 PNG（可配置）。  

---

## 3. 可视化选项（供选择）

### 选项 A（推荐基线）: `hist + ecdf + rank`
1. 直方图（整体形状与离散程度）。  
2. ECDF（尾部行为与阈值可读性好）。  
3. 排名曲线（best 到 worst 的坡度）。  

优点:
1. 信息密度高且稳定。  
2. 对分布比较最友好。  

缺点:
1. 不直接展示“复杂度-性能”关系。  

### 选项 B: `A + complexity_scatter`
1. 在 A 基础上增加复杂度代理分数 vs EPE 散点图。  

优点:
1. 可快速判断“更重是否更好”。  

缺点:
1. 复杂度仅是代理分数，不等价真实 FLOPs/Latency。  

### 选项 C: `full`（A + B + box）
1. 全量图组：`hist, ecdf, rank, complexity_scatter, box`。  

优点:
1. 一次性覆盖全视角。  

缺点:
1. 图较多，阅读成本高。  

---

## 4. 实施计划
1. 新增脚本 `code/nas/supernet_subnet_distribution.py`。  
2. 新增 wrapper `wrappers/run_supernet_subnet_distribution.py`。  
3. 新增轻量单测（采样唯一性、复杂度分数方向正确性）。  
4. 更新计划索引 `00_Plan_Index.md`。  

---

## 5. 运行建议（默认）
1. 首次建议: `num_arch_samples=512`。  
2. 若 GPU/时间充足: 提升到 `1024`。  
3. 图表模式建议先用选项 B（平衡信息量与诊断价值）。  
