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

## Phase 1 Findings (Accumulated During Execution)

*待开发过程中追加。*
