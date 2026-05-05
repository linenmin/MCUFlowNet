# Progress Log: search_hybrid_v1

## Session: 2026-05-05 (Planning Initialized)

### Status

- **Status**: planning complete for Phase 1, Phase 2-5 outlines only

### Actions taken

- 创建 `plan/search_hybrid_v1/` 目录
- 写 `task_plan.md`:
  - 整体 Goal 和 5 条 Cross-Phase Design Principles
  - 5 phase 总览表
  - Phase 1 详细拆分（5 个工作块: 1.1 旧 agent 残骸清理 / 1.2 Vela parser /
    1.3 监控指标补全 / 1.4 generation_metrics schema 升级 / 1.5 insights.md
    格式定义）
  - Phase 2-5 outline 占位
  - Open Questions 5 条
- 写 `findings.md`:
  - 3 条 Foundational Decisions (F-001, F-002, F-003)
  - F-DEP-001 标记 search_v3 已完成的可复用基础设施
- 写 `progress.md`（本文件）
- 在 `plan/search_v3/progress.md` 末尾追加 Final Status 行，宣告该计划冻结

### Key Pre-Phase Conclusions

来自之前对话的关键诊断结论（已在 findings.md 固化）:

1. NSGA-II 在 800 evals 双目标 Pareto 上不可战胜（HV=4.21 vs agent HV=3.89）
2. agent 真正强项是先验热启动、归纳总结、监督，不是中段插值
3. agent 在前 100 evals HV 领先 ~5%（best_fps 8.91 在 16 evals 内即被找到）
4. 当前 NSGA-II 监控指标严重不足，特别是 HV 完全没有记录
5. 旧 agent 系统的 micro_insight 同质化（771 条 63% 提 backbone, 31% 提 DB0），
   硬件信息从未真正进入决策回路

### Next Recommended Action

开始执行 Phase 1，建议顺序:

1. **1.1 旧 agent 残骸清理** —— 必须最先做，避免新代码污染
2. **1.2 Vela 层级数据 Python parser** —— 后续 Scientist 的依赖
3. **1.3 NSGA-II 监控指标补全** —— Supervisor 的依赖，工作量最大
4. **1.4 generation_metrics.csv schema 升级** —— 与 1.3 同步
5. **1.5 insights.md 文件格式定义** —— 写规范文档，无代码

### Open Questions Carried Into Phase 1 Execution

参见 `task_plan.md` 末尾的 Open Questions。开始 Phase 1 执行前不一定要解决，
但敏感性问题（特别是 HV 参考点）需要在 1.3 实现时决定。

---

## Session: 2026-05-05 (Phase 1.1 Complete)

### Status

- **Status**: Phase 1.1 (Old Agent Cleanup) complete

### Files modified

- `efnas/search/prompts.py`: 删除 `AGENT_C_SYSTEM`, `AGENT_D1_SYSTEM`,
  `AGENT_D2_SYSTEM`, `AGENT_D3_SYSTEM`；保留 `UNIVERSAL_WORLDVIEW`,
  `AGENT_A_SYSTEM`, `AGENT_B_SYSTEM`（Phase 2 重写）
- `efnas/search/agents.py`: 删除 `invoke_agent_d1/d2/d3`,
  `execute_verification_script`, `execute_candidate_check_script`,
  `filter_candidates_by_findings`, `_extract_topic_summary`；保留
  `invoke_agent_a`, `invoke_agent_b` 及共享辅助函数（`_summarize_history`,
  `_summarize_current_pareto`, `_summarize_pareto_dynamics`,
  `_summarize_epoch_metrics`, `_count_pareto_2d`, `_compute_pareto_front_df`,
  `_build_coverage_hint`）
- `efnas/search/eval_worker.py`: 删除 `_invoke_agent_c` 和
  `_read_per_layer_report`；`evaluate_single_arch` 不再产 `micro_insight`，
  该列对新评估始终为空字符串
- `efnas/search/coordinator.py`: 整体重写为过渡形态。删除 8 phase 状态机里的
  Phase 1 (Scientist 大反思)、Phase 2 (assumption 验证)、Phase 2b (finding 再
  验证)、Phase 5 子段 (filter_candidates_by_findings)。当前协调器只剩
  A → B → dedup → eval → record 的简化链。`__init__` 不再读
  `scientist_trigger_interval` 和 `assumption_confidence_threshold`
- `efnas/search/file_io.py`: 删除全部 assumption I/O
  (`read/write/append_assumptions`, `remove_assumption_by_id`,
  `get_next_assumption_id`)、findings registry I/O
  (`read/write_findings_registry`, `upsert_finding`, `count_findings`,
  `parse_findings`, `remove_finding_by_id`,
  `render_active_finding_hints`, `summarize_active_findings`,
  `read_findings`, `write_findings`, `_read_legacy_findings_*`)、
  `write_verification_script`。`_default_run_state` 删除
  `scientist_done` / `assumptions_evaluated` / `findings_revalidated` 三个
  阶段标记。`_is_valid_search_experiment_dir` 不再要求
  `assumptions.json` / `findings.json|md` 存在。`init_experiment_dir` 不再
  创建占位 JSON

### Tests modified or removed

- 删除（整体测试已删除的功能）：
  - `tests/test_agent_control_loop_refactor.py`
  - `tests/test_file_io_registry_and_run_state.py`
  - `tests/test_prompt_template_formatting.py`
  - `tests/test_search_dryrun.py`
- 修改（去掉对已删除 patch 目标的引用）：
  - `tests/test_search_coordinator_v2_metrics.py`：移除对
    `count_findings` 和 `read_assumptions` 的 mock，改为只验证
    `coverage_pct`、`findings_count=0`、`assumptions_count=0`、
    `rule_rejected=0`

### Verification

- `D:/Anaconda3/envs/tf_work_hpc/python.exe -m unittest tests.test_search_coordinator_v2_metrics tests.test_nsga2_baseline tests.test_run_nsga2_search_wrapper tests.test_eval_worker_command -v` → **18 tests OK**
- 全套 `unittest discover -s tests` → **153 tests, 1 unrelated error**
  - 唯一 error 是 `test_system_overview_pdf_export` 缺 `reportlab` 包（PDF 导出测试），
    与本次 cleanup 无关
- 符号存在性二次验证：所有要删的都不存在，所有要留的都存在
- 导入链验证：`efnas.search.{agents, coordinator, file_io, prompts,
  eval_worker}` + `efnas.baselines.nsga2_search` 全部 import 成功

### Behavioral Notes for Downstream Phases

- `history_archive.csv` 的 `micro_insight` 列对新评估永远为空字符串。
  Phase 1.2 不会改变这一点；Phase 1.2 引入的层级数据走独立的
  `analysis/layer_profile.json` 文件，不复用 micro_insight 字段
- `epoch_metrics.csv` 的 `findings_count` 和 `assumptions_count` 列保留为
  向后兼容字段，新写入永远是 0。Phase 1.4 做 schema 升级时会一起删除
  这两列
- `run_state.json` 中删除的三个阶段字段对旧实验目录是向后兼容的（read 时被
  `_default_run_state` 过滤掉，write 时不再写入）
- 当前 `efnas/search/coordinator.py` 处于过渡形态。Phase 2/3/4 会重写它
  以承载新 hybrid 架构（warm-start + scientist + supervisor）。在那之前，
  `wrappers/run_agentic_search.py` 仍可运行但只剩简化的 A→B→eval 闭环

### Next Action

进入 **Phase 1.2: Vela Layer Profile Python Parser**。任务是:
- 写 `parse_vela_layer_profile(vela_summary_path) -> List[Dict]`
- 每子网评估完成后持久化层级 profile 到
  `outputs/<exp>/dashboard/eval_outputs/run_<arch>/analysis/layer_profile.json`
- 在 Coordinator 加 `query_vela_for_arch(arch_code)` 接口
- 单元测试 + 集成测试

---

## Session: 2026-05-05 (Phase 1.2 Complete)

### Status

- **Status**: Phase 1.2 (Vela Layer Profile Python Parser) complete

### Files added

- `efnas/search/vela_parser.py`：新模块。导出
  - `parse_vela_layer_profile(per_layer_csv_path) -> List[Dict]`
  - `find_per_layer_csv(run_output_dir) -> Optional[str]`
  - `write_layer_profile(run_output_dir, layer_profile) -> str`
  - `parse_and_persist_layer_profile(run_output_dir) -> Optional[str]`
  - `query_vela_for_arch(exp_dir, arch_code) -> Optional[List[Dict]]`
  - `_extract_block_tag(raw_name) -> str` (内部辅助)
- `tests/test_vela_parser.py`：27 个单元/集成/smoke 测试

### Files modified

- `efnas/search/eval_worker.py`：在 `evaluate_single_arch` 评估成功后自动
  调用 `parse_and_persist_layer_profile(run_output_dir)`，把 Vela per-layer
  CSV 转成 `analysis/layer_profile.json`。出错只 log 不阻断主流程。
- `efnas/search/coordinator.py`：新增 `SearchCoordinator.query_vela_for_arch`
  方法，薄封装 `vela_parser.query_vela_for_arch(self.exp_dir, arch_code)`。
  Phase 3 Scientist agent 在 hardware grounding 阶段会显式调用。

### Key Design Choices

1. **Schema 列定位用索引而非 csv.DictReader**：Vela per-layer CSV 有重复
   的 `Network%` 列名（cycles 占比和 MACs 占比同名），DictReader 会合并。
   parser 直接用位置索引，最低需要 15 列。
2. **block_tag 是启发式短标签**：从 raw_name 里扫已知模式（H1Out 早于 H1、
   EB0/EB1/DB0/DB1、E0_/E1_ 融合标记、AccumResize/Add 尾部聚合）。匹配不
   到返回 "Other"，不强行猜测。
3. **持久化 JSON 而不只 in-memory**：让 Phase 3 Scientist 即使在 Vela
   tflite 已被 prune 的情况下仍能拿到层级数据；同时支持跨 run 复用。
4. **query 接口先读 JSON 再 fallback 到 CSV**：避免 Phase 1.2 之前的旧实验
   目录无法被 Phase 3 Scientist 查询。
5. **eval_worker 集成失败不阻断评估**：Vela 层级解析是 best-effort，
   失败仅 logger.exception，不影响主历史记录写入。

### Verification

- `tests.test_vela_parser`: **27/27** passed (含真实 search_v1 样本 smoke test)
- 联合套件 `test_vela_parser + test_search_coordinator_v2_metrics +
  test_nsga2_baseline + test_run_nsga2_search_wrapper +
  test_eval_worker_command + test_file_io_prune +
  test_file_io_resume_selection + test_llm_json_retry`：**48/48** passed
- 符号导入验证: `vela_parser` 全部 5 个公共函数 + 1 个内部辅助都存在；
  `SearchCoordinator.query_vela_for_arch` 方法存在

### Behavioral Notes for Downstream Phases

- 新评估自动产 `analysis/layer_profile.json`（约 30-40 层、几 KB），
  Phase 3 Scientist 直接读这个文件，零运行时解析开销
- 旧实验目录（无 layer_profile.json）的 query_vela_for_arch 会 fallback
  到 per-layer CSV 现场解析；这意味着 Phase 3 在跨实验做对照分析时也能
  拿到历史层级数据
- block_tag 的命名约定（"EB0"、"DB0"、"H1Out" 等）会成为 Phase 3
  Scientist prompt 的术语基础。后续如果搜索空间命名变化，需要同步更新
  `_BLOCK_TAG_PATTERNS`

### Next Action

进入 **Phase 1.3: NSGA-II 监控指标补全（HV / crowding / entropy / gap /
stagnation 等 8 大类）**。这是 Phase 4 Supervisor 的硬依赖。工作量是
Phase 1 里最大的一块（1-1.5 天），但完成后 NSGA-II baseline 自身在论文里
就更值钱（更详尽的搜索健康度报告）。

---

## Session: 2026-05-05 (Phase 1.3 + 1.4 Complete)

### Status

- **Status**: Phase 1.3 (NSGA-II 监控指标补全) + Phase 1.4 (schema 升级) 都
  完成。两块紧耦合所以一并交付。

### Files added

- `efnas/search/search_metrics.py`：纯函数模块，导出
  - 8 类指标算子: `hypervolume_2d`, `shannon_entropy`,
    `per_dim_gene_entropy`, `mean_crowding_distance_excluding_inf`,
    `largest_pareto_gap`, `stagnation_count`, `_compute_pareto_front_2d`
  - 聚合函数: `compute_full_generation_metrics(...)`
  - Schema 常量: `GENERATION_METRICS_COLUMNS` (37 列), `DEFAULT_HV_REF_EPE`
    = 5.5, `DEFAULT_HV_REF_FPS` = 3.0
- `tests/test_search_metrics.py`：30 个单元 + 集成测试

### Files modified

- `efnas/search/file_io.py`:
  - `append_epoch_metrics` 增加 `columns` 关键字参数 (Phase 1.4)
  - 抽出 `_LEGACY_EPOCH_METRICS_COLUMNS` 作为旧 schema 默认值
  - Migration 逻辑保持: 已存在文件的列与目标列不一致时按目标列 rewrite
- `efnas/baselines/nsga2_search.py`:
  - `_run_single_generation` 把 `next_population` 透给 `_record_generation_metrics`
  - `_record_generation_metrics` 调用 `search_metrics.compute_full_generation_metrics`
    并把 `columns=GENERATION_METRICS_COLUMNS` 传给 file_io

### Key Design Choices

1. **HV 参考点选 (5.5, 3.0)**：基于 search_v2_refactor_run2 实测数据，最差
   子网约 (EPE=5.0, FPS=3.5)，外推一格作为 ref。可由 `compute_full_generation_metrics`
   的 `ref_epe` / `ref_fps` 参数覆盖（写入 CSV 的 `hv_ref_epe`/`hv_ref_fps` 两列
   保留参考点用于事后分析）。
2. **保留 `epoch` 列名不重命名为 `generation`**：偏离原 task plan 的"重命名"
   决定 —— 重命名风险大（多个分析脚本和测试可能间接依赖 `epoch` 列名），收益
   小（语义已通过 docstring 说清）。文档化保留。
3. **保留 `findings_count` / `assumptions_count` 两列**：写 0，作为向后兼容
   字段。等 search_hybrid_v1 完整跑过一次再决定是否真的删除。
4. **Shannon 熵用 natural log**：和 log_2 只差常数，supervisor 比较时不影响
   方向判断；natural log 在数值稳定性上略好。
5. **rank1_saturation 基于"当前种群"而非"历史全量"**：种群是 NSGA-II 真正在
   做 crossover 的工作集；如果种群本身已经全在第一前沿，crossover 就是在自己
   跟自己玩 —— 这是 Phase 4 Supervisor 真正关心的塌缩信号。
6. **stagnation 用严格改进计数**：v_t < running_best 才算改进 (decrease)，
   等值不算。这避免"反复达到同一最优值"被误判为持续改进。
7. **schema 字段类型容错**：history_df 里 epe/fps 列可能是字符串（旧实验
   遗留），`compute_full_generation_metrics` 里全部走 `pd.to_numeric(errors=
   "coerce")` + `dropna`，缺数据不会让整代指标计算失败。

### Verification

- `tests.test_search_metrics`: **30/30** passed
- 联合套件 `test_search_metrics + test_vela_parser +
  test_search_coordinator_v2_metrics + test_nsga2_baseline +
  test_run_nsga2_search_wrapper + test_eval_worker_command +
  test_file_io_prune + test_file_io_resume_selection + test_llm_json_retry`：
  **78/78** passed
- 全套 `unittest discover`: **210 tests, 1 unrelated error** (仍是无关的
  `reportlab` 缺包；和 Phase 1.1/1.2 一致)
- API 验证: `search_metrics` 全部 10 个公共导出名都在；`file_io.append_epoch_metrics`
  签名为 `(exp_dir, metrics, *, columns=None)`

### Behavioral Notes for Downstream Phases

- 新 NSGA-II 实验的 `epoch_metrics.csv` 有 37 列；旧实验的 11 列文件**不会
  被破坏**（只有当 NSGA-II 在旧目录上 resume 才会触发 migration）
- 第一代 (epoch=0) 的 stagnation 全是 0，HV 改进率是空字符串 ——
  Phase 4 Supervisor 的 prompt 必须显式处理"序列还不够长"的边界情况
- `hv_ref_epe` / `hv_ref_fps` 两列写入每行：让事后分析时能验证 HV 比较的
  参考点一致；如果未来调整参考点，旧记录仍可解释
- `mean_crowding_distance` 在前沿 ≤ 2 点时返回 0.0（不是空字符串）——
  Phase 4 Supervisor 区分"前沿小所以无 interior"和"前沿大但拥挤"时要看
  pareto_count 列辅助判断
- Phase 3 Scientist 也可以读 `epoch_metrics.csv` 这 37 列做反思辅助；不是
  Supervisor 专属

### Next Action

进入 **Phase 1.5: insights.md 文件格式定义**（写规范文档，无代码）。这是
Phase 1 的最后一块，工作量小，1-2 小时即可完成。完成后 Phase 1 整体收尾，
进入 Phase 2 (Warm-Start Agent) 详细设计。

---

## Session: 2026-05-05 (Phase 1.5 Complete + Phase 1 Closeout)

### Status

- **Status**: Phase 1.5 完成 → Phase 1 整体收尾。

### Phase 1.5 Files added

- `efnas/search/insights.py` (140 行, 9 个公共函数 + 2 个内部正则常量)
- `efnas/search/templates/insights_template.md` (空骨架, 注释里说明三条硬约束)
- `tests/test_insights.py` (24 个测试)

### Phase 1.5 Files modified

- `efnas/search/file_io.py`:
  - `init_experiment_dir` eagerly 创建 `metadata/insights.md` 从模板拷贝
  - 新增内部 `_load_insights_template()` 辅助函数

### Phase 1.5 Key Design Choices

**最重要的决策: 偏离原 task plan 的"完整字段模板"，改为最小契约**:

1. **只锁三个机器解析不变量**: heading 格式 + status enum + ID 形式
2. **正文完全自由形式**: 没有 Pattern / Confidence / Hardware grounding /
   Generator hint 等任何必填字段
3. **退役不需要移区段**: status='retired' 标签自带过滤能力

理由: Scientist agent 是真正的 LLM, 给它字段 schema 就会被迫填字段——没东西
说时硬凑信息密度反而稀释正文价值。最小契约让 Scientist 自由表达, 又给下游
解析提供足够的机器抓手。

其他细节决策:
- ID `I-` 前缀强制, 后续允许字母数字短横线 → 支持 `I-001` 数字风格也支持
  `I-EB0-DB1` 语义风格
- `next_insight_id` 跳过非数字 ID → 语义化 ID 不影响下一个数字编号
- 模板的 `---` 分隔线之上是注释区, 之下是 Scientist 内容区 (但不强制)
- CRLF 换行被正则 `\s*$` 吞掉, Windows 友好

### Phase 1.5 Verification

- `tests.test_insights`: **24/24** passed
- 联合套件: `test_insights + test_search_metrics + test_vela_parser +
  test_search_coordinator_v2_metrics + test_nsga2_baseline +
  test_run_nsga2_search_wrapper + test_eval_worker_command +
  test_file_io_prune + test_file_io_resume_selection`: **101/101** passed

### Phase 1 整体收尾汇总

| 子模块 | 状态 | 测试 | 主要产出 |
|---|---|---|---|
| 1.1 旧 agent 残骸清理 | ✅ | 删 4 + 修 1 | 5 个文件清理 |
| 1.2 Vela 层级 parser | ✅ | 27/27 | `efnas/search/vela_parser.py` |
| 1.3 NSGA-II 监控指标 | ✅ | 30/30 | `efnas/search/search_metrics.py` |
| 1.4 schema 升级 | ✅ | (含在 1.3) | `append_epoch_metrics` 加 columns |
| 1.5 insights 格式 | ✅ | 24/24 | `efnas/search/insights.py` + 模板 |

**Phase 1 累计新增代码**:
- 3 个新模块: `vela_parser.py`, `search_metrics.py`, `insights.py`
- 1 个模板: `templates/insights_template.md`
- 3 个新测试文件: `test_vela_parser.py`, `test_search_metrics.py`,
  `test_insights.py`
- 修改: `eval_worker.py`, `coordinator.py`, `file_io.py`,
  `nsga2_search.py`, `prompts.py`, `agents.py`

**Phase 1 累计删除代码**:
- 4 个旧测试 (`test_agent_control_loop_refactor.py`,
  `test_file_io_registry_and_run_state.py`,
  `test_prompt_template_formatting.py`, `test_search_dryrun.py`)
- `agents.py` 里的 D-1/D-2/D-3 + Agent C + filter_candidates_by_findings
- `coordinator.py` 里的 Phase 1/2/2b 调度
- `file_io.py` 里的 assumption/finding I/O + write_verification_script
- `prompts.py` 里 4 套 D/C 系列 prompt
- `eval_worker.py` 里 `_invoke_agent_c` 和 `_read_per_layer_report`

**Phase 1 完成判定 (Success Criteria)**: 全部 7 项达成，详见 task_plan.md
末尾 Success Criteria 区段。

### Open Questions Status

回顾 task_plan.md 里的 5 个 Open Questions:

1. **HV 参考点 (5.5, 3.0)**: Phase 1.3 已敲定; 写入 CSV 的 `hv_ref_epe`/
   `hv_ref_fps` 列保留可追溯性
2. **Scientist 触发频率 K**: **未决**, Phase 3 详细设计时再敲定
3. **Warm-start 多样性约束**: **未决**, Phase 2 详细设计时再敲定
4. **Supervisor 与 Scientist 同步还是错开**: **未决**, Phase 4 详细设计时
5. **insights.md 字段结构化程度**: ✅ 已敲定 (走最小契约), 见 1.5 决策记录

### Next Action

进入 **Phase 2: Warm-Start Agent**。Phase 2 是最小最独立的 LLM agent 角色,
预计 1-2 天:
- 单次 LLM 调用 (generation 0 之前)
- 输入: 搜索空间语义描述 + 多样性要求 + 目标方向
- 输出: 50 个 arch_code 作为 NSGA-II 初始种群
- NSGA-II runner 加 `external_initial_population` hook
- 多样性 sanity check
- Ablation: NSGA-II only vs +warm-start (HV @ 100 evals 对比)

下次先讨论 Phase 2 的几个关键决策点:
- Prompt 工程: 如何精确表达"覆盖 EPE-extreme / FPS-extreme / 中段"
- 多样性 sanity check 的判据 (覆盖率 / 熵阈值 / 极端点存在性)
- 如果 LLM 生成的种群多样性不达标, 是 fallback 到 NSGA random init 还是
  retry / partial fill

---

## Session: 2026-05-05 (Phase 2 Complete)

### Status

- **Status**: Phase 2 (Warm-Start Agent) 完成. 偏离原 outline 里的"多样性
  sanity check 阈值 + 不达标 retry"，改为"全局信息 + 角色定位 + 硬约束,
  diagnostics 只观察不门控"，让 agent 自己做策略取舍.

### Files added

- `efnas/search/warmstart_agent.py` (200 行):
  - `invoke_warmstart_agent` (单次 LLM 调用 + 失败兜底)
  - `compute_warmstart_diagnostics` (合法性 + entropy + rationale)
  - `save_warmstart_diagnostics` (落盘)
  - `warmstart_pipeline` (端到端封装供 wrapper 调用)
- `tests/test_warmstart_agent.py` (15 个测试)

### Files modified

- `efnas/search/prompts.py`:
  - `UNIVERSAL_WORLDVIEW` 重写: 反映 hybrid 团队 (NSGA-II + Warmstart +
    Scientist + Supervisor); 删除 D-1/D-2/D-3 + Agent C 描述; 加入
    NSGA-II 算子细节 (uniform per-gene crossover 90%, per-gene 1/11 mutation)
  - 删除 `AGENT_A_SYSTEM`, `AGENT_B_SYSTEM`
  - 新增 `WARMSTART_AGENT_SYSTEM` (含 `{population_size}` 占位符, 3816 字符)
- `efnas/baselines/nsga2_search.py`:
  - `__init__` 增加 `external_initial_population` kwarg
  - `_run_single_generation` generation 0 优先消费外部种群
  - 新增 `_consume_external_initial_population` 方法 (验证 + 去重 + partial
    random fill)
- `wrappers/run_nsga2_search.py`:
  - 加 `--enable_warmstart` 和 `--warmstart_role` CLI flags
  - `--resume` 时 warmstart 被跳过 (避免重复污染 gen 0)
  - 调用 `warmstart_pipeline` 一行端到端

### Files deleted (legacy agentic stack 整体清退)

- `efnas/search/agents.py` (Agent A/B invoke + helpers)
- `efnas/search/coordinator.py` (transitional 形态, Phase 3 会重新写)
- `wrappers/run_agentic_search.py` (legacy 入口)
- `tests/test_search_coordinator_v2_metrics.py` (测试已删类)

### Key Design Choices

1. **不门控 diagnostics**: 即使 LLM 输出多样性差或全部非法, NSGA-II 都不会被
   阻断 — 兜底是 partial random fill。这让 ablation 数据更诚实地反映"该次
   LLM 决策的实际后果".
2. **告诉 agent 算子细节**: prompt 显式给出 NSGA-II 的 uniform per-gene
   crossover (90%) + 1/11 mutation. 这是 agent 自主决策需要的关键信息 ——
   它能据此推断"基因池多样性对后续 crossover 探索能力的意义".
3. **rationale 字段强制有但内容自由**: agent 必须输出 rationale, 但内容
   自由形式. 落盘到 diagnostics, Phase 5 ablation 时可读. 没东西说就简短
   陈述, 不强求信息密度.
4. **--resume 跳过 warmstart**: 避免续跑时 LLM 调用产生不一致初始种群.
5. **同步整体清退 legacy 栈**: 既然 Phase 2 应当"删 A 重写 B", 干脆把
   coordinator + agentic wrapper + 对应测试一起删, 让代码结构干净. Phase 3
   Scientist + Phase 4 Supervisor 会用新接口写, 不依赖任何残留.
6. **强制恰好 50 个**: prompt 里硬规定输出数量, 工程层校验通过 partial
   random fill 兜底. 不强迫 retry —— LLM 偶尔少给几个不应触发昂贵重调.

### Verification

- `tests.test_warmstart_agent`: **15/15** passed
- 联合套件 `test_warmstart_agent + test_insights + test_search_metrics +
  test_vela_parser + test_nsga2_baseline + test_run_nsga2_search_wrapper +
  test_eval_worker_command + test_file_io_prune + test_file_io_resume_selection
  + test_llm_json_retry`: **116/116** passed
- 全套 `unittest discover`: **248 tests, 1 unrelated error** (仍是 reportlab
  缺包，与 Phase 1.1/1.2/1.3/1.5 一致)
- API 验证: warmstart_agent 全部 4 个公共导出存在; wrapper `--enable_warmstart`
  + `--warmstart_role` 在 help 中; `WARMSTART_AGENT_SYSTEM.format()` 工作正常

### Open Questions Resolved in Phase 2

参考 task_plan.md Open Questions:
- **#3 Warm-start 多样性约束**: 已敲定 — 走"不规定 + 只观察 diagnostics"路线,
  让 agent 自己做策略取舍

仍未决:
- **#2 Scientist 触发频率 K**: Phase 3 详细设计时再定
- **#4 Supervisor 与 Scientist 同步还是错开**: Phase 4 详细设计时再定

### Next Action

进入 **Phase 3: Scientist Agent** —— 系统创新点核心. 下次会议先讨论:
- 触发频率 K (建议 3, 但需要确认: NSGA-II 16 代里反思 5-6 次合理吗?)
- Scientist 工作流的两阶段 (Phase A 架构归纳 → Phase B 硬件 grounding 检验)
  的 prompt 设计
- Coordinator 怎么调度 Scientist 的 Vela 查询 + Python 验证脚本执行
- 失败兜底: Scientist 返回空 / 输出格式破坏怎么办
- insights.md 的备份策略 (每次 Scientist 调用前 snapshot 成 insights.md.gen{N})
