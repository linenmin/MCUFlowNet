# Fold 版本诊断与 CPU Eval 计划
更新时间: 2026-02-17

## 0. 目标与范围
本文基于以下输入进行诊断和规划:
1. fold 版本日志（epoch 53~58）。
2. 当前训练与评估代码（`supernet_trainer.py`、`supernet_eval.py`、`fc2_dataset.py`、`dataloader_builder.py`）。
3. 已切换为 folder-based FC2 数据加载（`train_dir`/`val_dir`）。

核心目标:
1. 先验证“脚本化 CPU eval”在 HPC 上是否可行（不改 yaml）。
2. 按优先级列出潜在问题，给出每个问题的优化选项与优缺点。
3. 审视当前评估指标，明确哪些要保留/降权/删除，并引入排名稳定性指标（含 Kendall tau）。

---

## 1. 先做: HPC 上脚本化 CPU Eval 可行性验证
当前已提供脚本入口: `EdgeFlowNAS/wrappers/run_supernet_eval.py`，支持 CLI 覆盖配置与 `--cpu_only`。

建议先在 HPC 做两次同配置对照（GPU vs CPU），用时与结果一起看:

```bash
cd /data/leuven/379/vsc37996/test/MCUFlowNet/EdgeFlowNAS

# 1) GPU baseline
/usr/bin/time -f "gpu_eval_wall=%E maxrss=%MKB" \
python wrappers/run_supernet_eval.py \
  --config configs/supernet_fc2_180x240.yaml \
  --checkpoint_type last \
  --experiment_name edgeflownas_supernet_fc2_180x240_run1_folder \
  --base_path /data/leuven/379/vsc37996/test/MCUFlowNet \
  --train_dir Datasets/FlyingChairs2/train \
  --val_dir Datasets/FlyingChairs2/val \
  --batch_size 32 \
  --bn_recal_batches 8

# 2) CPU eval
/usr/bin/time -f "cpu_eval_wall=%E maxrss=%MKB" \
python wrappers/run_supernet_eval.py \
  --config configs/supernet_fc2_180x240.yaml \
  --checkpoint_type last \
  --experiment_name edgeflownas_supernet_fc2_180x240_run1_folder \
  --base_path /data/leuven/379/vsc37996/test/MCUFlowNet \
  --train_dir Datasets/FlyingChairs2/train \
  --val_dir Datasets/FlyingChairs2/val \
  --batch_size 32 \
  --bn_recal_batches 8 \
  --cpu_only
```

判定标准:
1. `status=ok` 且生成 `supernet_eval_result_last.json`。
2. CPU 用时是否小于单个训练 epoch（当前约 9~11 分钟）。
3. CPU/GPU 的 `mean_epe_12` 差异是否在可接受波动区间（先用经验阈值 3%~5%）。

---

## 2. 关键现象（来自日志）
1. fold 版本下 `train_samples=17785`，`val_samples=640`。这与 FlyingChairs2 的 train/val 规模一致，640 本身不是异常。
2. 恢复到 epoch 53 后，`mean_epe_12` 稳定在约 10.3~11.2，显著高于旧阶段 0.6~4.2 区间。
3. `micro_batch_size` 从 4 改到 8 后，`mean_epe_12` 基本同量级，未体现明确收益。
4. `fairness_gap` 持续为 0.00，说明采样公平性机制正常，但该指标区分度接近 0。

---

## 3. 按优先级的问题与优化选项

## P0-1 评估口径切换导致指标“断层”
问题描述:
历史阶段曾受 data list 与 fallback 影响，当前切换到真实 `val_dir` 后，指标分布发生跳变。旧 best/early-stop 语义与新口径不再可比。

当时实现考虑:
先保证训练可继续，允许 fallback 与旧口径共存，便于快速迭代。

现状缺点:
1. 断点恢复时容易误读“退化/提升”。
2. 历史最佳 checkpoint 与当前 eval 定义脱节。

优化选项:
1. 选项 A: 口径切换后强制新实验名，并重置 early-stop 状态。
优点: 实现简单、语义清晰、与已有 `reset_early_stop_on_resume` 兼容。
缺点: 历史曲线不连续，需要额外对照分析。

2. 选项 B: 在日志和 checkpoint meta 写入 `eval_protocol_version`，恢复时做兼容检查。
优点: 可追溯性强，避免“误恢复”。
缺点: 需要补一层协议管理字段。

3. 选项 C: 口径切换后固定做一次“桥接评估”（同 checkpoint 在旧/新口径都评一次）。
优点: 可量化断层幅度。
缺点: 旧口径已逐步弃用，维护成本较高。

## P0-2 当前 eval 噪声较大，直接驱动 early-stop 风险高
问题描述:
每个 epoch 的 supernet eval 是“每个 arch 一次 val 随机 batch + BN 重估随机 batch”。采样方差可能较大。

当时实现考虑:
为了控制评估时长，采用小样本快速代理指标。

现状缺点:
1. `mean_epe_12` 对 batch 采样敏感。
2. early-stop 可能受短期噪声触发。

优化选项:
1. 选项 A: 固定 eval batch 集（固定索引或固定随机种子序列）。
优点: 方差显著下降，可重复性提升。
缺点: 可能过拟合固定小集合。

2. 选项 B: 每个 arch 使用多 batch 求均值（例如 3~5）。
优点: 指标更稳健。
缺点: 评估时间线性增加。

3. 选项 C: 采用 EMA 平滑 early-stop 输入（不改原始记录，仅改停止判定）。
优点: 改动小，对抗瞬时抖动有效。
缺点: 会延迟对真实退化的响应。

## P0-3 同步评估阻塞训练，CPU 资源未被利用
问题描述:
当前训练线程 epoch 结束后同步执行 eval，GPU 训练等待 eval 完成，CPU 大内存资源基本空闲。

当时实现考虑:
单会话串行流程最稳，避免并发 checkpoint 读写一致性问题。

现状缺点:
1. 增加 epoch 墙钟时间。
2. 难利用 HPC 的 32GB CPU RAM 做并行评估。

优化选项:
1. 选项 A: 每 N 个 epoch 才评估一次（例如 N=2 或 5）。
优点: 最小改动，立即提速。
缺点: 反馈频率下降，early-stop 粒度变粗。

2. 选项 B: 训练后异步启动独立 CPU 进程做 eval（读取 `supernet_last.ckpt` 快照）。
优点: 可并行，训练不阻塞。
缺点: 若 CPU eval > epoch 时间，会堆积；需要任务去重与“只保留最新评估”策略。

3. 选项 C: 双层评估，epoch 内快速 proxy，间隔做完整 CPU eval。
优点: 兼顾速度与可靠性。
缺点: 指标体系更复杂，需要定义两层指标职责。

## P1-4 `eval_every_epoch` 配置未生效（配置-实现不一致）
问题描述:
配置中存在 `eval.eval_every_epoch`，但训练循环中未使用该字段，实际行为是每个 epoch 都 eval。

当时实现考虑:
默认每轮评估便于观测，先实现主流程。

现状缺点:
1. 配置误导。
2. 无法通过配置降低评估频率。

优化选项:
1. 选项 A: 实装 `eval_every_epoch`，并在非评估轮跳过 early-stop 更新。
优点: 语义直观，改动小。
缺点: 非评估轮无验证反馈。

2. 选项 B: 改为 `eval_interval_steps`（按 step 控制）。
优点: 对不同 `steps_per_epoch` 更稳定。
缺点: 配置复杂度增加。

## P1-5 训练日志中的 `loss`/`grad_norm` 是“最后一次累积步”快照
问题描述:
当前日志记录的是最后一个 arch + 最后一个 micro-batch 的瞬时值，不是 epoch 平均。

当时实现考虑:
最少日志开销，快速拿到可见信号。

现状缺点:
1. 容易误判“梯度爆炸”或“训练变稳”。
2. 难与 eval 波动做因果对应。

优化选项:
1. 选项 A: 记录 epoch 平均/中位数/分位数（例如 P50/P90）。
优点: 统计意义更强。
缺点: 需要累积缓存，日志更长。

2. 选项 B: 同时保留快照与平均值。
优点: 兼顾调试与趋势判断。
缺点: 指标条目变多。

## P1-6 BN 重估成本与收益尚未量化
问题描述:
每个 arch 评估前做 `bn_recal_batches` 次前向，成本高；收益是否足够未量化。

当时实现考虑:
超网不同子网共享 BN 统计，重估可减少评估偏差。

现状缺点:
1. 评估开销大。
2. 可能成为异步 CPU eval 的主要瓶颈。

优化选项:
1. 选项 A: 先做消融（0/2/4/8 批）观察 rank 一致性。
优点: 数据驱动决定配置。
缺点: 需要额外实验。

2. 选项 B: 使用缓存或低频重估（隔轮重估）。
优点: 可显著降时延。
缺点: 统计陈旧可能引入偏差。

## P2-7 `fairness_gap` 长期 0.00，监控价值有限
问题描述:
在 strict fairness 机制下该值几乎固定，难提供额外诊断信息。

当时实现考虑:
用于验证采样器是否破坏公平分配。

现状缺点:
1. 信息增量低。
2. 容易造成“指标看起来健康”的错觉。

优化选项:
1. 选项 A: 保留但降权，不用于停止/选模。
优点: 兼顾可追溯与简洁。
缺点: 仍占日志空间。

2. 选项 B: 替换为每 block/option 的频次偏差分布摘要。
优点: 更细粒度，异常可见。
缺点: 指标更复杂。

## P2-8 缺少“排名稳定性”指标（建议增加 Kendall tau）
问题描述:
当前只看均值与标准差，无法直接回答“子网排序是否稳定”。

当时实现考虑:
先用简单标量驱动 early-stop，降低系统复杂度。

现状缺点:
1. 可能出现均值接近但排序大幅变化。
2. 对 NAS 选模价值不足。

优化选项:
1. 选项 A: 增加 Kendall tau（当前排序 vs 参考排序）。
优点: 对排序一致性敏感，适合 NAS。
缺点: 需要定义参考排序来源。

2. 选项 B: 增加 Spearman rho + Top-K overlap@K（K=3/5）。
优点: 易解释，补充 tau 信息。
缺点: 指标数量增加。

3. 选项 C: 仅增加 Top-K overlap。
优点: 与最终选模目标直接相关。
缺点: 忽略全排序信息。

## P2-9 val 样本 640 导致统计波动相对更明显
问题描述:
val 集较小，且含随机 crop，评估方差天然偏高。

当时实现考虑:
遵循 FC2 官方 train/val 划分，保证协议一致。

现状缺点:
1. 单轮波动较明显。
2. 更依赖固定采样与重复评估策略。

优化选项:
1. 选项 A: 固定中心裁剪用于 eval。
优点: 降低随机性。
缺点: 与训练增强分布有偏差。

2. 选项 B: 保留随机裁剪但固定随机种子序列。
优点: 可复现且保留一定分布多样性。
缺点: 仍有采样偏差风险。

## P3-10 切换 micro-batch 的收益需要“吞吐-质量”联合评估
问题描述:
日志显示 mbs=4 与 mbs=8 在短窗口 `mean_epe_12` 接近，收益不明显。

当时实现考虑:
通过 micro-batch 适配显存并提升吞吐。

现状缺点:
1. 仅看 loss/EPE 难判断最佳 mbs。
2. 缺少 samples/sec 与显存峰值对照。

优化选项:
1. 选项 A: 统一跑 mbs 网格（4/8/16）并记录吞吐、显存、评估指标。
优点: 可量化选型。
缺点: 需要额外机时。

2. 选项 B: 固定 mbs=8 作为当前保守值，先解决评估体系问题。
优点: 降低变量数，推进更快。
缺点: 可能错过最佳吞吐点。

---

## 4. 评估方式与指标删改建议
保留:
1. `mean_epe_12`，作为主要误差尺度指标。
2. `std_epe_12`，作为子网间离散度指标。

降权:
1. `fairness_gap` 仅保留诊断，不参与 early-stop 或选模。

建议新增:
1. `kendall_tau`，比较“当前 per-arch 排序”与“参考排序”的一致性。
2. `spearman_rho`，作为排名相关的补充。
3. `top_k_overlap@3/@5`，直接反映候选头部子网稳定性。
4. `eval_confidence`（例如 bootstrap CI 或多次 eval 的标准误），用于判断本轮指标可信度。

建议替换 early-stop 判据:
1. 现有: 只用 `mean_epe_12`。
2. 建议: 主判据 `mean_epe_12_ema`，辅判据 `kendall_tau` 下限（例如 tau < 0.5 连续若干轮不保存 best）。

---

## 5. 推荐执行顺序
1. 先在 HPC 跑第 1 节的 GPU/CPU eval 对照命令，拿到真实墙钟时间与结果差异。
2. 立即实现 `eval_every_epoch` 生效，先支持降频评估（低风险快收益）。
3. 增加固定 eval 样本机制，降低 `mean_epe_12` 噪声。
4. 再引入排名指标（Kendall tau + Top-K overlap）并灰度接入 early-stop。
5. 视第 1 步结果决定是否做异步 CPU eval（仅当 CPU 评估时长可控时推进）。

---

## 6. 验收标准
1. 在真实 val（640）下，连续 10 个 epoch 的 `mean_epe_12` 波动显著收敛（相较当前策略）。
2. `eval_every_epoch` 配置实装且可通过日志验证生效。
3. CPU eval 方案有明确可行性结论（可并行/不可并行，并附时间证据）。
4. 排名稳定性指标可输出并用于分析（至少产出 tau 和 top-k overlap）。
