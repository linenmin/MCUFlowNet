# Task Plan: Hybrid LLM-Augmented NSGA-II for Hardware-Aware NAS

## Goal

在 V3 supernet 上构建一个 **NSGA-II 主导 + LLM agent 增强** 的多目标 NAS 系统。
NSGA-II 仍然是核心搜索引擎，LLM 通过三个互不重叠的角色提供增强：

1. **Warm-start**: 利用 LLM 的架构先验和搜索空间认知生成多样化初始种群
2. **Scientist**: 从已评估的 (arch_code, EPE, FPS) 数据中归纳出人类可读的设计规则，
   并通过 Vela 层级数据做硬件 grounding
3. **Supervisor**: 监控 NSGA-II 健康指标，在停滞或多样性塌缩时调整 mutation_prob
   （动作空间严格受限）

系统**不**用 LLM 过滤或拒绝候选，**不**强制 if-then findings 形式，**不**让 LLM
和 NSGA-II 竞争 Pareto 中段填充。Vela 硬件数据是 LLM 洞察的**侧面佐证**，不是
驱动信号。

## Cross-Phase Design Principles

每个 phase 的所有决策都要回头对照这 5 条原则：

1. **NSGA-II 是承重墙，不是被取代项**: 任何修改不得改变 NSGA-II 在没有 agent
   介入时的行为。所有 agent 都是可关闭的旁路。
2. **每个 agent 角色必须有独立 ablation**: 进入论文之前必须证明它单独贡献了
   HV / 规则数 / 收敛速度的可测量增量。
3. **Vela 是 agent 的查询服务，不是驱动信号**: Coordinator 不主动喂层级 cycles
   给 agent，agent 想看才给。
4. **insights 是自由形式产出物，不是控制流**: 不进入 NSGA-II 算子，不做硬过滤。
   论文里它是产出，不是机制。
5. **Supervisor 行动空间有硬上限**: mutation_prob 调整必须在
   `[0.5×, 1.5×] × current` 内；任何超界尝试视为 prompt 失败、回退默认。

## Current Phase

Phase 2 ✅ Complete (2026-05-05). Ready to enter Phase 3.

## Phases Overview

| Phase | Name | Core Deliverable | LLM Involved? |
|---|---|---|---|
| 1 | Foundations & Cleanup | 监控指标补全 + Vela parser + 旧 agent 残骸清理 + insights.md 格式 | ❌ 纯 Python |
| 2 | Warm-Start Agent ✅ | 单次 LLM 生成种子种群 + NSGA-II init hook + 不门控 diagnostics | ✅ 1 次/run |
| 3 | Scientist Agent | 周期性大反思 + Vela grounding + insights.md 演化 | ✅ K 代/次 |
| 4 | Supervisor Agent | mutation_prob 受限调整 + rollback 机制 | ✅ K 代/次 |
| 5 | Validation & Paper Artifacts | 四组 ablation + 设计规则验证 + 论文图表 | ❌ 离线分析 |

## Phase 1: Foundations & Cleanup

**核心约束**: 这一 phase 完全不涉及任何 LLM 调用，是纯 Python 工程。但它是后面
所有 phase 的硬依赖。

### 1.1 旧 Agent 残骸清理 ✅ Complete (2026-05-05)

- [x] 删除 `efnas/search/agents.py` 里的 `invoke_agent_c`, `invoke_agent_d1`,
  `invoke_agent_d2`, `invoke_agent_d3`, `execute_verification_script`,
  `execute_candidate_check_script`, `filter_candidates_by_findings`
- [x] 删除 `efnas/search/coordinator.py` 里的 `_evaluate_pending_assumptions`,
  `_revalidate_findings`, `_execute_scientist_macro_reflection`,
  `_resolve_rule_script_path` 及其调度循环
- [x] 删除 `efnas/search/file_io.py` 里 assumption / finding I/O 全套
  （保留 `history_archive.csv`, `epoch_metrics.csv` 相关）
- [x] 删除 `efnas/search/prompts.py` 里 `AGENT_C_*`, `AGENT_D1_*`,
  `AGENT_D2_*`, `AGENT_D3_*` 全部模板
- [x] 保留 `AGENT_A_SYSTEM`, `AGENT_B_SYSTEM` 暂不动（Phase 2 重写 B；A 在
  Phase 2 收工后一起删）
- [x] 删除 `eval_worker.py` 里的 `_invoke_agent_c` 和 `_read_per_layer_report`
- [x] 删除/修改对应的 obsolete 测试 (4 个删除 + 1 个修改)
- [x] 跑 NSGA-II baseline 相关 18 tests + 全套 153 tests 验证零回归
  （唯一 error 是无关的 `reportlab` 缺包问题）

**理由**: 后续 Phase 在 `agents.py` 和 `coordinator.py` 动刀时，留着旧代码容易
让新代码复用错的旧接口。旧 prompt 模板的 if-then findings 措辞会污染 Phase 3
Scientist 自由形式 prompt 的设计。

### 1.2 Vela 层级数据 Python Parser ✅ Complete (2026-05-05)

- [x] 实现 `efnas/search/vela_parser.py::parse_vela_layer_profile(per_layer_csv_path) -> List[Dict]`
  - 输入: Vela 工具产出的 `*_per-layer.csv` 路径
  - 输出: `[{block_tag, tflite_op, nng_op, sram_bytes, peak_pct, cycles,
    cycles_pct, util_pct, macs, macs_pct, raw_name}, ...]`
- [x] 每个评估完成的子网持久化层级 profile 到
  `outputs/<exp>/dashboard/eval_outputs/run_<arch_code>/analysis/layer_profile.json`
  （在 `eval_worker.evaluate_single_arch` 评估成功后自动调用
  `parse_and_persist_layer_profile`）
- [x] 在 Coordinator 加查询接口
  `SearchCoordinator.query_vela_for_arch(arch_code) -> Optional[List[Dict]]`
  （薄封装 `vela_parser.query_vela_for_arch`，按需读 JSON，找不到 fallback 回
  CSV）
- [x] 旧 Agent C 的 LLM 调用路径已在 1.1 删除
- [x] 单元 + 集成测试 (`tests/test_vela_parser.py`)：27/27 通过
  - block_tag 提取（H1Out 优先级、E0/E1 融合标记、AccumResize 精确 tag、
    Other/Unknown fallback）
  - 解析合成 CSV (3 行覆盖典型场景)
  - 错误处理（缺文件、列数不足、单行畸形）
  - find_per_layer_csv 两种目录布局
  - write_layer_profile JSON 回环
  - parse_and_persist_layer_profile 端到端
  - query_vela_for_arch 优先 JSON / fallback CSV / 空返回
  - **真实样本 smoke test**：解析 search_v1 buggy_backup 实际 CSV，确认
    EB0/DB0/H1Out/AccumResize2 等已知 block_tag 都被正确识别，util_pct
    全部落在 [0, 100] 区间

### 1.3 NSGA-II 监控指标补全 ✅ Complete (2026-05-05)

新增 `efnas/search/search_metrics.py`（纯函数模块，导出
`compute_full_generation_metrics` + 8 类指标算子 + `GENERATION_METRICS_COLUMNS`
schema）。`efnas/baselines/nsga2_search.py::_record_generation_metrics` 改成
调用聚合函数并把 `next_population` 透给它（用于种群层面熵和 rank-1 saturation
计算）。30 个新单元测试全部通过；联合 78 tests 零回归（NSGA-II / file_io /
vela_parser / coordinator 测试都过）。

- [x] **Hypervolume (HV / 超体积)** —— `hypervolume_2d(pareto_points,
  ref_epe=5.5, ref_fps=3.0)`，闭式测试覆盖单点 / 三点楼梯 / 参考点过滤 / 无序
  输入。
- [x] **HV 改进率 (hv_improvement_rate_3gen)** —— `(HV_t - HV_{t-3}) / 3`，序列
  长度 < 4 时返回空字符串（避免 None vs NaN 之争）。
- [x] **平均拥挤距离 (mean_crowding_distance)** —— NSGA-II 标准 crowding
  distance，去除 ±inf 边界后取算术平均；前沿点数 ≤ 2 返回 0.0。
- [x] **每维基因熵 (gene_entropy_dim_0 .. gene_entropy_dim_10)** —— 当前种群每维
  取值分布的 Shannon 熵 (natural log)，11 个独立列。空种群全 0；malformed
  arch_code 自动跳过。
- [x] **停滞代数三个独立计数 (stagnation_best_epe / stagnation_best_fps /
  stagnation_hv)** —— 自上次该指标朝改进方向真正变化以来的代数；EPE 用
  decrease，FPS 和 HV 用 increase。
- [x] **最大 Pareto gap (largest_gap_fps_low / fps_high / epe_low / epe_high)**
  —— 前沿按 FPS 排序后相邻 FPS 间距最大处的两个端点；前沿 < 2 点返回空字符串。
- [x] **重复率趋势 (duplicate_rate, duplicate_rate_3gen_avg)** ——
  `duplicate_rate = duplicates / population_size`；3-gen 滑动均值用
  metrics_history 最近 3 行 + 当前值。
- [x] **第一前沿饱和度 (rank1_saturation)** —— 当前种群里第一非支配前沿大小 /
  种群大小（基于 history_archive 查 (epe, fps) 后做局部非支配排序）。
- [x] 单元测试 + 边界 case：30 个测试覆盖闭式 HV / Shannon 熵的 log(2) log(3)
  基准 / 拥挤距离三点共线 / Pareto gap 三点找最大 / stagnation 改进+停滞+边界 /
  compute_full_generation_metrics 端到端 + schema 一致性

**理由**: 当前 NSGA-II 只记 6 个指标，最关键的 HV 没记。supervisor 决策的关键
不是单个数值而是趋势和多维健康度。这一项 ablation 价值独立 —— 即使后续 LLM
全失败，更详尽的搜索健康度报告本身就让 NSGA-II baseline 在论文里更值钱。

### 1.4 generation_metrics.csv Schema 升级 ✅ Complete (2026-05-05)

- [~] **保留** `epoch` 列名（不重命名为 `generation`）—— 决策修订: 重命名风险大
  且收益小（多个分析脚本和测试可能间接依赖 `epoch` 列名），收益不抵风险。改为
  在文档里说明 `epoch` 列在 NSGA-II 语境下表示 generation。
- [~] **保留** `findings_count`, `assumptions_count` 两列做向后兼容，新评估写 0。
  实际删除待 search_hybrid_v1 完整跑过一次后再考虑。
- [x] 新增 21 个字段（与 1.3 列表完全对齐），最终 `GENERATION_METRICS_COLUMNS`
  共 37 列。
- [x] 旧 search_v2 实验的 `epoch_metrics.csv` 不动（保留为只读历史）。
- [x] `file_io.append_epoch_metrics` 增加可选 `columns` 参数：NSGA-II 传新 schema，
  老的 transitional coordinator 不传该参数沿用 legacy schema (11 列)；migration
  逻辑会把已有文件的 schema rewrite 到调用方指定的 schema，缺失字段填空。
- [x] 文档化: `efnas/search/search_metrics.py` 顶部 docstring 列出 8 大类指标，
  `_LEGACY_EPOCH_METRICS_COLUMNS` 和 `GENERATION_METRICS_COLUMNS` 在文件里对照。

**理由**: schema 是数据契约，必须显式版本化。Phase 4 Supervisor 直接消费这个
CSV，schema 不定就没法写 prompt。

### 1.5 insights.md 文件格式定义 ✅ Complete (2026-05-05)

**最终决策: "最小契约"原则** —— 偏离原 task plan 的"完整字段模板"，改为只锁
三个机器可解析的不变量，正文完全自由（不强制 Pattern/Confidence/Hardware
grounding/Generator hint 等任何字段）。这样 Phase 3 Scientist 不会被字段
schema 逼着硬凑信息密度。

**机器解析的硬约束（仅此三条）:**
1. 每条 insight 用三级标题: `### I-{ID} ({status}): {一行标题}`
2. status 三选一: `active` / `retired` / `under_review`（大小写敏感）
3. ID 必须以 `I-` 开头, 后续只允许字母数字和短横线; 一旦分配后稳定不变

正文（heading 之后到下一条 heading 之前）**完全自由形式**: 没有必填字段、
没有字段顺序、可贴 Python 代码、可画 ASCII 表、可中英混写、可留空。

实施明细:
- [x] 实现 `efnas/search/insights.py` (140 行):
  - `INSIGHT_HEADING_RE` 正则 (CRLF 兼容)
  - `VALID_STATUSES = {"active", "retired", "under_review"}`
  - `parse_insights(md_text)` -> `List[Dict[id, status, title, body]]`
  - `list_active_insights / list_active_ids` 过滤
  - `next_insight_id(md_text, prefix="I-", width=3)` 返回下一个 sequential
    ID; 跳过非数字风格的 ID (例如 `I-EB0-DB1`)
  - `validate_id` / `find_insight_by_id` / `count_by_status`
- [x] 创建 `efnas/search/templates/insights_template.md`:
  - 注释里完整说明三条硬约束
  - `---` 之上是 Coordinator 区域, 之下是 Scientist 区域
  - 不强制任何字段
- [x] 更新 `file_io.init_experiment_dir` eagerly 创建 `metadata/insights.md`
  从模板拷贝
- [x] 单元测试 `tests/test_insights.py` (24 个):
  - parse_insights: 空 / 只有 header / 单条 / 多条 / 不匹配的 ### 子标题归入
    body / status 大小写敏感 / Python 代码块保留 / 语义化 ID (I-EB0-DB1) /
    CRLF 换行
  - next_insight_id: 空文件 / sequential / 跳过非数字 ID / max+1 而非 count+1 /
    自定义宽度
  - validate_id: 6 个合法 + 8 个不合法形式
  - find_insight_by_id, count_by_status
  - 模板文件可解析为空; init_experiment_dir 写出可解析的 insights.md

**Phase 4 Supervisor 解析路径**: 调用 `parse_insights(md_text)` 拿到 list，
filter status == "active"，直接用 title 和 body 作为软建议; 不试图理解 body
内部结构。

**版本化备份**: 每次 Scientist 调用前 Coordinator 把当前 `insights.md` 复制
为 `insights.md.gen{N}` (这部分逻辑放在 Phase 3 实现, Phase 1 只把格式定下来)。

### Success Criteria ✅ All passed (2026-05-05)

Phase 1 完成判定:

1. ✅ `efnas/search/agents.py` 里只剩 Agent A、B 旧 prompt（待 Phase 2 重写），
   没有 D 系列、Agent C
2. ✅ `efnas/search/file_io.py` 里没有 assumptions/findings 相关函数，但
   history、generation_metrics 相关函数完整
3. ✅ NSGA-II baseline 的 `_record_generation_metrics` 调用
   `compute_full_generation_metrics`, 写出 37 列 (含 HV / crowding / entropy
   / gap / stagnation 等 8 大类指标), 通过 30 个单元测试
4. ✅ `parse_vela_layer_profile` + 真实 Vela 样本 smoke test 通过 (27 个测试),
   block_tag 启发式正确识别 EB0/DB0/H1Out/AccumResize2 等已知块
5. ✅ Coordinator 的 `query_vela_for_arch(arch_code)` 方法存在; 优先读
   layer_profile.json, fallback 到 per-layer CSV
6. ✅ `insights_template.md` 落盘到 `efnas/search/templates/`,
   `init_experiment_dir` eagerly 创建 metadata/insights.md, 24 个测试覆盖
   parse_insights / next_insight_id / validate_id 等
7. ✅ 单元测试全过 (Phase 1 累计):
   - Phase 1.1 cleanup: 18 NSGA 相关测试 + 联合 153 全套 (1 unrelated error)
   - Phase 1.2 vela_parser: 27 (含真实样本 smoke)
   - Phase 1.3+1.4 search_metrics: 30
   - Phase 1.5 insights: 24
   - **联合 101 个新增/相关测试零回归**

**实际工作量**: 一个工作天内完成全部 5 个子模块。原估 1.5-2 天, 实际比预期快
（搜索基础设施已经就绪 + vela_parser 用了真实样本快速验证）。

## Phase 2: Warm-Start Agent ✅ Complete (2026-05-05)

**最终设计原则: "全局信息 + 角色定位 + 硬约束, 不规定策略"** —— 偏离原 outline
里的"多样性 sanity check 阈值"，改为只观察不门控。让 agent 看到下游全景 (NSGA-II
uniform per-gene crossover, per-gene 1/11 mutation, 后续 Scientist + Supervisor
存在) 后自己做策略取舍.

### 2.1 NSGA2SearchRunner external_initial_population hook ✅

- [x] `__init__` 增加 `external_initial_population: Sequence[str]` kwarg
- [x] `_run_single_generation` 在 generation 0 时优先消费外部种群
- [x] `_consume_external_initial_population` 方法: 验证合法性 + 跨 batch
  去重 + 不足时调 `_sample_unique_random_arches` partial random fill
- [x] 工程兜底: 任何环节出错 (LLM 不响应 / 输出 0 个合法) 自动 fallback
  random init, NSGA-II 不会因 warmstart 失败而挂

### 2.2 Warmstart Agent 模块 + Prompt ✅

- [x] 新增 `efnas/search/warmstart_agent.py`:
  - `invoke_warmstart_agent(llm, *, population_size, role)` — 单次 LLM 调用
  - 失败兜底返回 `{rationale: "", arch_codes: [], raw_response: None}`
- [x] 新 prompt `prompts.WARMSTART_AGENT_SYSTEM`:
  - `UNIVERSAL_WORLDVIEW` 已重写: 反映 hybrid 团队 (NSGA-II + Warmstart +
    Scientist + Supervisor); 删除旧的 D-1/D-2/D-3 + Agent C 描述
  - Role section 显式说明: "你的策略选择由你自己决定, 我不会替你想"
  - 给出 NSGA-II 算子细节 (uniform per-gene crossover 90%, per-gene 1/11
    mutation), 让 agent 能推断"基因池多样性对 crossover 的意义"
  - 硬约束只锁: 恰好 50 条、合法 arch_code 范围、JSON 格式
- [x] **同步删除** `efnas/search/agents.py`, `efnas/search/coordinator.py`,
  `wrappers/run_agentic_search.py`, `tests/test_search_coordinator_v2_metrics.py`
  (legacy 代码; Phase 3 Scientist + Phase 4 Supervisor 会重新写)

### 2.3 Diagnostics (观察, 不门控) ✅

- [x] `compute_warmstart_diagnostics(arch_codes, *, search_space, ...)`:
  - returned_count / valid_count / invalid_count /
    duplicate_within_batch / unique_valid_count
  - per_dim_entropy (11 维 Shannon, 基于 unique valid)
  - rationale + llm_model + timestamp 落盘做审计
- [x] `save_warmstart_diagnostics(exp_dir, diag)` 写入
  `metadata/warmstart_diagnostics.json`
- [x] **不阻止流程**: 即使诊断显示多样性差或全部非法, 仍把可用部分交给
  NSGA-II partial fill。Phase 5 ablation 时把这些诊断和最终 HV 关联做
  相关分析

### 2.4 Wrapper 集成 + 测试 ✅

- [x] `wrappers/run_nsga2_search.py` 加两个 CLI flag:
  - `--enable_warmstart`: 在 generation 0 之前调用 warmstart_pipeline
  - `--warmstart_role` (默认 "warmstart_agent"): LLM 路由 role 名
- [x] `--resume` 时 warmstart 被忽略 (避免重复污染 generation 0)
- [x] `warmstart_pipeline(llm, exp_dir, ...)` 一行端到端调用:
  invoke → compute diagnostics → save → return valid arch_codes
- [x] `tests/test_warmstart_agent.py`: 15 个测试覆盖
  - invoke_warmstart_agent: 正常解析 / LLM 抛错 / 非 dict 响应 /
    arch_codes 字段非 list / 空白条目 strip
  - compute_warmstart_diagnostics: 全合法 / 混合非法+重复 / 空输入
  - save_warmstart_diagnostics: JSON 回环
  - warmstart_pipeline: 端到端 (mocked LLM) / LLM 失败兜底
  - NSGA2SearchRunner._consume_external_initial_population: 全合法 /
    partial fill / 跳过非法 / 跨 batch 去重

### Ablation 设计 (Phase 5 跑)

- (a) 纯 random init NSGA-II
- (b) `--enable_warmstart` LLM warm-start NSGA-II
- 跑多次取均值, 关注 HV @ 100 evals 差距
- 副产品: `warmstart_diagnostics.json` 里的 per-dim entropy / unique_valid_count
  和最终 HV 做相关分析, 看 LLM 策略选择是否与最终结果相关

## Phase 3: Scientist Agent (Outline)

**待 Phase 1, 2 完成后详细展开。** 当前仅记录核心要点:

- 每 K=3 代调用一次（待定）
- 两阶段工作流:
  - 阶段 A: 从 (arch_code, EPE, FPS) 自由形式归纳洞察
  - 阶段 B: 对每条洞察主动查询 Vela 数据做 grounding
- 可选写 Python 验证片段（沙箱执行）
- 输出: 修订后的 insights.md（追加 + 修订 + retire）
- Ablation: insights 在 holdout 验证集上的命中率

## Phase 4: Supervisor Agent (Outline)

**待 Phase 1, 2, 3 完成后详细展开。** 当前仅记录核心要点:

- 每 K 代调用一次（与 Scientist 同步或错开）
- 输入: 全套监控指标 + Scientist 最新 insights
- 输出: 紧 schema JSON（默认 `no_change`）
- 动作空间限制: `mutation_prob` 调整上限 `[0.5×, 1.5×]`
- Rollback 机制: K 代内 HV 未改善则恢复
- Ablation: rule-based 自适应 vs LLM Supervisor

## Phase 5: Validation & Paper Artifacts (Outline)

**待 Phase 1-4 完成后详细展开。** 当前仅记录核心要点:

- 四组 ablation 完整跑:
  - (a) NSGA-II only
  - (b) NSGA-II + Warm-start
  - (c) NSGA-II + Warm-start + Scientist
  - (d) NSGA-II + Warm-start + Scientist + Supervisor
- HV、Pareto count、收敛速度、insights 命中率对比图
- 论文图表生成

## Dependencies

- 输入: `plan/search_v3` 已完成的 V3 evaluator、V3 wrapper、search_space
  可配置、多 GPU 子进程绑定、CPU prefetch（参见
  `plan/search_v3/progress.md` 末尾"Final Status"）
- V3 supernet checkpoint:
  `outputs/supernet/edgeflownas_supernet_v3_fc2_172x224_run1_archparallel`
- NSGA-II V3 config: `configs/nsga2_v3.yaml`
- NSGA-II baseline runner: `efnas/baselines/nsga2_search.py`

## Open Questions (需要后续讨论)

1. **HV 参考点**: 当前提议 `(EPE_ref=5.5, FPS_ref=3.0)`，敏感性检查后是否需要
   调整？
2. **Scientist 触发频率 K**: 提议每 3 代一次，但 16 代总长下是 5 次反思，是否
   足够？
3. **Warm-start 多样性约束**: 如何在 prompt 里精确表达"覆盖 EPE-extreme /
   FPS-extreme / 中段"，避免 LLM 全产先验最优解？
4. **Supervisor 与 Scientist 同步还是错开**: 同步省一次 LLM 调度复杂度，错开
   让 Supervisor 能消费 Scientist 最新输出，待定。
5. **insights.md 是否需要更结构化**: 当前提议自由 markdown + 三种 status / 三种
   grounding 枚举。是否需要在 status 之外加 confidence 数值字段（避免完全自由
   描述）？

## References

- CoLLM-NAS (arXiv 2509.26037): 双 LLM Navigator + Generator 协作 NAS 范式
- A-NSGA-II (arXiv 1305.4947): 用 generation-wise improvement rate +
  crowding diversity 调 mutation rate
- HA-NSGA-II (Sage 2025): RL-based Adaptive Parameter Self-Tuning，
  本系统的 LLM Supervisor 是它的 LLM 替代版
- LLM-as-Meta-Optimizer 综述 (Springer AI Review 2025): LLM 作为参数控制
  策略生成器的范式
