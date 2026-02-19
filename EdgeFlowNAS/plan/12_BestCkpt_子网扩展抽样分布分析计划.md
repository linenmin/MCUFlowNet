# 12 BestCkpt 子网扩展抽样分布分析计划
更新时间: 2026-02-19

## 0. 目标与边界
1. 解释 `supernet_best.ckpt` 的保存判据与恢复语义。
2. 在 best checkpoint 下做更大规模子网抽样，联合统计 `EPE + FPS + SRAM peak`。
3. 严格拆分两阶段流程：先产出分析数据文件，再由独立脚本输出 PNG 图表。
4. 保持 EdgeFlowNAS 现有工程风格，方便后续维护。

---

## 1. BestCkpt 判据（已确认）
1. 最优权重依据 `mean_epe_12`（越小越好）。
2. 仅当 `current < best - early_stop_min_delta` 才会覆盖 best。
3. `--reset_early_stop_on_resume` 会重置早停状态，恢复训练后重新比较 best。

代码位置:
- `MCUFlowNet/EdgeFlowNAS/code/engine/early_stop.py`
- `MCUFlowNet/EdgeFlowNAS/code/engine/supernet_trainer.py`

---

## 2. 本轮确认选项
1. 可视化方案: `Option A + epe_vs_fps`。
2. 核心图表:
- `epe_hist.png`
- `fps_hist.png`
- `sram_hist.png`
- `epe_rank_curve.png`
- `epe_vs_fps_scatter.png`

说明:
- 你已明确优先关注 `EPE vs FPS`。
- SRAM 仍保留统计与图表，用于确认是否确实“基本一致”。

---

## 3. 已落地实现
### 3.1 分析阶段（只产数据）
已改造:
- `MCUFlowNet/EdgeFlowNAS/code/nas/supernet_subnet_distribution.py`

当前行为:
1. 先评估抽样子网 EPE（沿用原 supernet eval 逻辑）。
2. 可选执行 Vela 指标采集（每个子网导出固定 arch TFLite，再跑 Vela）。
3. 只输出数据文件，不在该脚本里绘图。

输出文件:
- `analysis/records.csv`
- `analysis/ranking_by_epe.csv`
- `analysis/vela_metrics.csv`
- `analysis/summary.json`
- `analysis/sampled_arch_pool.json`

新增参数（核心）:
- `--enable_vela`
- `--vela_mode`
- `--vela_optimise`
- `--vela_limit`
- `--vela_rep_dataset_samples`
- `--vela_float32`
- `--vela_keep_artifacts`
- `--vela_verbose_log`

### 3.2 绘图阶段（只读数据）
已新增:
- `MCUFlowNet/EdgeFlowNAS/code/nas/supernet_subnet_distribution_plot.py`

当前行为:
1. 读取 `analysis/records.csv`。
2. 输出 Option A + `epe_vs_fps` 的 PNG 图表。
3. 产出 `plot_manifest.json`。

### 3.3 Wrapper
已改造/新增:
- `MCUFlowNet/EdgeFlowNAS/wrappers/run_supernet_subnet_distribution.py`（分析）
- `MCUFlowNet/EdgeFlowNAS/wrappers/run_supernet_subnet_distribution_plot.py`（绘图）

### 3.4 单测
已更新:
- `MCUFlowNet/EdgeFlowNAS/tests/test_supernet_subnet_distribution.py`

覆盖点:
1. 抽样唯一性与可复现。
2. complexity 方向一致性。
3. FPS 换算稳定性。
4. 指标 summary 的有效/无效计数。

---

## 4. 推荐运行模板
### 4.1 先跑分析（含 Vela）
```bash
python wrappers/run_supernet_subnet_distribution.py \
  --config configs/supernet_fc2_180x240.yaml \
  --checkpoint_type best \
  --num_arch_samples 256 \
  --enable_vela \
  --vela_optimise Size \
  --vela_mode basic \
  --top_k 12 \
  --output_tag best_256_vela
```

### 4.2 再跑绘图
```bash
python wrappers/run_supernet_subnet_distribution_plot.py \
  --analysis_dir <run_dir>/analysis
```

---

## 5. 后续可选优化（未做）
1. Vela 并行子进程（当前是串行，优先保证稳定性）。
2. 增加 `epe_vs_sram` 与 `fps_vs_sram`（若后续转 Option B）。
3. 为 Vela 失败样本输出更细分错误分类。
