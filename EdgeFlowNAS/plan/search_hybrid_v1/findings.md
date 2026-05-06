# Findings: search_hybrid_v1

> 这个文件累积"开发过程中确认的关键事实和决策"，区别于 task_plan.md（计划）和
> progress.md（执行日志）。每条 finding 应该是事后看仍然可被引用的稳定结论。

## Foundational Decisions (Pre-Phase-1)

### F-001: NSGA-II 是承重墙，agent 是增强层

- **决策**: 系统主搜索引擎是 NSGA-II，不是 agent team
- **依据**:
  - 实测数据: search_v2_refactor_run2 vs nsga2_v2_run1 在 800 evals 下 NSGA
    HV=4.21 完爆 agent HV=3.89（详见
    `outputs/search_compare_refactor_run2_vs_nsga2_20260411/`）
  - 同一份数据显示 agent 在前 100 evals HV 领先 ~5%（warm-start 优势）
  - LLM 不擅长 Pareto 中段几何插值（NSGA crowding 的核心能力）
- **影响**: agent 在系统里只承担三个不与 NSGA 竞争的角色（warm-start / 归纳 /
  监督）

### F-002: 旧 finding 升格机制必须删除，不修补

- **决策**: 删除整套 `assumption→eval_script→confidence>=0.95→finding→hard_filter`
  管道
- **依据**:
  - 49 epoch 跑下来只升格 2 条 finding（A07, A16），ROI 极低
  - 升格规则要求 confidence ≥ 0.95，而 EPE 是非线性涌现指标，几乎所有架构
    级洞察都跨不过这个阈值 → 升格通道天然偏向硬件类规则
  - 硬件类规则又因为搜索空间共享同一硬件指纹（DB0 ~33% util）而同质化严重
  - if-then 二值规则形式不适合双目标 Pareto，本身就只能产 FPS-side 洞察
- **影响**: insights.md 改成自由形式 markdown，不再有"升格"步骤

### F-003: Vela 是 LLM 洞察的侧面佐证，不是驱动信号

- **决策**: Coordinator 不主动喂 Vela 数据给 LLM，LLM 想看才查
- **依据**:
  - 旧 Agent C per-arch LLM 调用产出 771 条 micro_insight，63% 提 backbone、
    31% 提 DB0，Util 中位数 33%，**结构化重复无效信息**
  - 搜索空间设计本身就把硬件下限保住了，逐层指标不会有过分缺陷 →
    "用硬件维度否定子网"反而适得其反（用户原话）
  - LLM 应该先归纳架构模式，再选择性查 Vela 找硬件解释
- **影响**: Phase 1 实现 Vela parser + Coordinator 查询接口；Phase 3 Scientist
  的工作流分两阶段，阶段 B 才查 Vela

## Inherited Dependencies

### F-DEP-001: V3 NSGA-II 基础设施已就绪（来自 search_v3）

- 来源: `plan/search_v3` (final status: complete, 2026-05-05)
- 可直接复用:
  - `efnas/engine/supernet_v3_evaluator.py`: V3 fixed-subnet evaluator
  - `efnas/nas/supernet_subnet_distribution_v3.py` + 对应 wrapper
  - `efnas/baselines/nsga2_search.py` 已支持 `search_space_module` 配置
  - `efnas/search/eval_worker.py` 支持 per-worker `CUDA_VISIBLE_DEVICES`
  - `configs/nsga2_v3.yaml`: V3 NSGA-II baseline config
- V3 supernet checkpoint:
  - no-distill: `outputs/supernet/edgeflownas_supernet_v3_fc2_172x224_run1_archparallel`
  - distill: `outputs/supernet/edgeflownas_supernet_v3_fc2_172x224_run1_archparallel_distill`
- 默认搜索目标: no-distill checkpoint (best inherited FC2 metric)

## Phase 5 Findings (Ablation 实测)

### F-101: Action space 的"形状"决定 supervisor 行为, 不是 prompt

- **发现**: 5-lever (mutation_prob, crossover_prob, per_dim_multiplier,
  tournament_size, reseed_bottom_pct) 在 11D 离散小空间下只有 2 个真正有锐度
  (mutation_prob, per_dim_multiplier)。supervisor 在停滞期会被迫挪 reseed
  ("最后一个能动的滑块"), 即使前沿模型有完整 NSGA-II 知识。
- **依据**: d_v1 (旧 prompt) HV=7.504, d_v2 (新 prompt) HV=7.515, d_v3 (8 lever
  + 新 prompt) HV=7.524。提升主要来自 **lever 数量 (+0.0099)** 而非 prompt
  修复 (+0.0105 是初次修复硬伤的量, 不可累加)。
- **影响**: 后续设计任何 LLM-controlled 算法时, 必须先确认 action space 在目标
  空间下**至少有 3-4 个有效杠杆**, 而不是只罗列一堆参数。

### F-102: Identity-default lever 是公平 ablation 的方法论判据

- **决策**: 任何 lever 只要 default 值让算法行为还原 baseline (e.g.
  `local_search_pareto_neighbors=0` 即不触发 local search), 就是干净的
  ablation lever。不需要"该机制本来就在 baseline 代码里"。
- **依据**: `mutation_neighborhood_bias=0` 的语义就是"uniform pick between
  alternatives" = 隐性的 baseline NSGA-II 行为。给 supervisor 这个 lever 是
  "提供偏离 baseline 的开关", 不是"赋予新算法能力"。
- **影响**: search_hybrid_v1 8 lever 全部满足 identity-default 判据, group a
  在 default 下行为与原 baseline NSGA-II 完全一致, group d 才能控制偏离。
- **不适用项**: `parent_pool_source` 改成 `history_pareto` 时需要 history
  archive 已积累足够规模 (至少 ≥ pop_size 的 rank-1 集才有意义), gen 0/1 切换
  没有效果。

### F-103: Prompt 里的诊断处方块 = 隐性 prior 注入

- **发现**: 即使写 "DIAGNOSTIC SIGNALS (诊断材料, 非动作处方)", 列条目本身就构成
  prior。"rank1_saturation > 0.8: crossover 在自己跟自己玩" 这种贬义 framing
  让 agent 把 NSGA-II 正常收敛诊断成病态 → 倾向干预。
- **决策**: prompt 只描述 raw column 名 + I/O 契约, 不预设"什么算异常"。
- **影响**: commit `b3d8e49` 砍掉 supervisor `DIAGNOSTIC SIGNALS` + Scientist
  B-2 `WHAT TO DO PER INSIGHT`。同一原则应用到任何后续 LLM-as-controller
  prompt 设计。

### F-104: Frozen_dims 释放 mutation 预算给活跃维 — 间接 Budget 优化

- **发现**: dim 7 (H1) 在搜索中段 entropy → 0 (H1=0 是 hardware-Pareto 最优, 真
  收敛)。继续在 dim 7 跑 mutation 是浪费概率预算。frozen_dims=[7] 让其他活跃维
  分到这部分 mutation rate。
- **依据**: d_v3 gen 5 supervisor 选 frozen_dims=[7] + local_search=5,
  dim 7 entropy 之后保持 0-0.098 (frozen 生效), 同时 dim 9 没有完全塌缩
  (gen 15: 0.227 vs d_v2 0.000)。
- **影响**: 后续若搜索空间含某些维度先验知道是收敛 → frozen 可作为干净的
  "不加 mutation 但允许 crossover 重组" 的开关。

### F-105: Memetic local search 在 1-flip 邻域上有 cache hit 副作用

- **发现**: d_v3 gen 11 supervisor 主动关掉 `local_search_pareto_neighbors=0`,
  rationale 写 "local search is causing the algorithm to repeatedly evaluate
  similar architectures"。
- **机制**: 1-flip 邻居枚举在搜索后期大概率撞 history_archive 已评估子网, 当前
  实现走 random fallback, 但 fallback 也会撞 (剩下未探索区域几乎都是 dominated
  arch)。最终 K 个 slot 中只有少数是真正新增多样性。
- **影响**: 待评估是否在 `_generate_pareto_neighbor_offspring` 里改成 K-flip
  (K≥2) 邻域, 或者从 history_pareto 第二、三层 (rank-2/3) 取邻居。这是实现层
  的优化, 不影响 lever 设计哲学。

### F-106: Scientist 在 group c 是 passive observer, 实际等价 group b + 调用开销

- **发现**: c 组 Scientist 输出 insights.md 但**没有下游消费者** (supervisor 是
  d 才有的)。NSGA-II 主循环不读 insights.md。
- **结论**: c vs b 的 HV 差 0.0014 = 纯 RNG/LLM warmstart 随机性, 不是
  Scientist 的贡献。
- **影响**: 下次单 seed ablation 若要省 budget, 可只跑 a/d 两组 (用 d 内部的
  Scientist 调用代替单独的 c 组)。或者在 paper 里诚实地把 b/c 合并成 "+
  warmstart" 一档。

### F-107: 旧 prompt 的 E0/E1 方向标错传染 Scientist hardware narrative

- **发现**: 早期 `UNIVERSAL_WORLDVIEW` 写 E0 `0=7x7 stride-2`, 实际
  `V3_BLOCK_SPECS` 是 `0=3x3 Conv`。Scientist gen5 I-001 直接信任 prompt 写
  "extreme FPS subnets employ E0=0 (7x7 stride-2 stem). Vela query confirms
  E0 cycles ~483k", 把 3x3 conv 的 cycles 数字硬解读成"7x7 优化得很好"。
- **教训**: agent 的 hardware grounding 信任度受 prompt 标签直接影响。**任何
  search-space 表都必须从 ground truth 代码 (e.g. V3_BLOCK_SPECS) 自动同步**,
  避免手写 prompt 与代码漂移。
- **影响**: 后续若搜索空间扩展, 应当从 `efnas/nas/search_space_v3.py` 自动生成
  prompt 的 search-space 段, 而不是手维护 prompt 副本。

## Phase 1 Findings (Accumulated During Execution)

*待开发过程中追加。*
