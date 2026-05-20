# Progress Log: search_hybrid_v2

## Session: 2026-05-20 (Planning Initialized)

### Status

- **Status**: Phase 1 Audit 准备就绪, 已有完整 v1 trajectory 数据 + 部分
  prompt 审计结论 (来自本次对话上一轮的分析)

### Actions taken

- 创建 `plan/search_hybrid_v2/` 目录
- 写 `task_plan.md`: 5 phase 切分 (Audit → Prompt 重写 → 模型评估 →
  d_v4 重跑 → 结果分析)
- 写 `findings.md`: 9 条 findings 已 pre-filled
  - F-DATA-001: 4 组撞 HV ceiling 7.524 (high)
  - F-PROMPT-001: agent 不知道是光流任务 (high)
  - F-PROMPT-002: warmstart 无覆盖目标 (medium)
  - F-PROMPT-003: Scientist insights 没被 Supervisor 真正消费 (high)
  - F-PROMPT-004: Supervisor budget-blind (medium)
  - F-PROMPT-005: warmstart 不知 teacher arch_code (low)
  - F-PROMPT-006: Supervisor reseed 诱惑 (medium)
  - F-LLM-001: LiteLLM 调用层 OK 但可调优 (low)
  - F-LLM-002: Gemini 3 不接受 temperature (high)
  - F-MODEL-001: 候选升级模型清单 (medium)
  - F-DESIGN-001: c 组 ablation 等价于 b (low)
  - F-DATA-002: 待 Phase 1.4 补预算切片数据 (medium)
- 写 `progress.md` (本文件)

### Pre-Phase 1 Conclusions (来自本次对话上一轮)

1. v1 4 组单 seed 收敛点 (a=7.5217, b=7.5245, c=7.5259, d_v3=7.5244) 全部
   在 ±0.005 HV 内, 噪声占优
2. LLM warmstart 在 gen 0 给 +1.5 HV 大领先, 但 a 在 gen 10 反超 b, 全部 4
   组撞 HV ceiling
3. d_v3 best_epe (4.0230) 是全场最优, budget efficiency 也最佳
   (gen 8 = 450 evals 达 HV=7.5, a 要 gen 10 = 550 evals)
4. LiteLLM 调用层无 bug, 但 Gemini 3 系列服务端限 temperature=1.0
5. agent 不知道任务是光流估计, 也不知道 supernet teacher / 数据集 / 分辨率
6. supervisor 在 d_v3 后期 (gen 11/14) 仍在挪 reseed, 是 budget-blind 决策

### Next Recommended Action

开始执行 Phase 1.1 (LLM 调用层审计) + 1.2 (Agent 输入信息完整性审计):

1. 详细读 `efnas/search/warmstart_agent.py` / `scientist_agent.py` /
   `supervisor_agent.py` 三个 agent 的 user_message 构造逻辑, 找出每个
   agent 实际拿到什么数据
2. 把每个 agent 缺失的关键信息列成 Phase 2 重写 checklist
3. 完成 Phase 1.4 数据驱动诊断 (50/100/200/400 evals 切片 HV)

### Open Questions Carried Into Phase 1 Execution

参见 task_plan.md 末尾 Key Questions, 重点:

1. 领域先验进 prompt 后, warmstart gen 0 HV 能否从 7.07 → 7.20+? (Phase 4
   实证回答)
2. Opus 4.7 / GPT-5 单次调用比 Gemini 3.1 Pro 贵 5-10×, HV 提升 0.005 值不值?
3. Supervisor reseed 诱惑用方案 A (commit lever) 还是 B (hard rule) 治?

### Verification baseline (for Phase 2-4)

- v1 d_v3 final HV: **7.5244**
- v1 d_v3 best_epe: **4.0230**
- v1 d_v3 budget-to-HV=7.5: **450 evals (gen 8)**

v2 d_v4 在以上任一指标取得 measurable improvement (HV +0.005 或 best_epe -0.005
或 budget -50 evals) 即可视为 v2 升级成功. 任何一项倒退即触发 Phase 5
方案降级 (退回 Gemini 3.1 Pro + 不引入领域先验, 用 v1 原版本写论文).

---

## Files modified this session

- 新增 `plan/search_hybrid_v2/task_plan.md`
- 新增 `plan/search_hybrid_v2/findings.md`
- 新增 `plan/search_hybrid_v2/progress.md`

---

## Session: 2026-05-20 (Prompt 逐个过审 6/6 完成)

### Status

- **Status**: 6 个 system prompt v2 草稿全部定稿 (✅), 落盘到 `prompts_draft.md`.
  下一步进入代码落地.

### Actions taken

按 search 流程先后顺序, 与用户逐个过审 6 个 prompt:

| # | Prompt | 关键决策 |
|---|---|---|
| 1 | UNIVERSAL_WORLDVIEW | 加 TASK SEMANTICS + OPTIMIZATION OBJECTIVE; 删错误的 trade-off 段; 5→8 lever 更新; 词数基本持平 |
| 2 | WARMSTART_AGENT_SYSTEM | 用户诊断: 原"关键"段中 *"任一维全同值 crossover 永远碰不到"* 警告 = 指挥 LLM 走最大均匀采样 = d_v3 per-dim entropy 接近最大的元凶. 全段删, 极简版词数 -65% |
| 3 | SCIENTIST_STAGE_A_SYSTEM | Stage A/B 对比 4 行 → 1 行; 5 种操作枚举 → 1 行精简; body 示例段全删; # CONSTRAINTS 重写成单行; 词数 -38% |
| 4 | SCIENTIST_STAGE_B1_SYSTEM | 保守减 ~12% (用户选 2c/3b 保留 OUTPUT 长 example + QUERY 5 条规则); 砍工程层兜底说明 + 末段哲学填充 |
| 5 | SCIENTIST_STAGE_B2_SYSTEM | INPUT 的 Vela query 结果 + verification 执行结果 的多行 JSON example 压缩为字段名清单; 其他保留; 词数 -21% |
| 6 | SUPERVISOR_AGENT_SYSTEM | 加 budget_progress 输入 (F-PROMPT-004 修复); 删 2 条哲学/工程兜底句; 8 lever schema 完整保留; 词数 -4% |

### Phase 1 Audit 隐式完成

按原 task_plan.md 设计, Phase 1 包含 1.1-1.5 多个子任务. 实际本次会话用
"逐个过审 prompt" 方式同时覆盖了:

- Phase 1.1 LLM 调用层审计: ✅ 已查 (llm_client.py 调用方式 OK, F-LLM-001/002 已定稿)
- Phase 1.2 Agent 输入完整性审计: ✅ 已查 (6 个 prompt 走读, F-PROMPT-001~006)
- Phase 1.3 Agent 输出契约审计: ✅ 已查 (走读 OUTPUT FORMAT 和机器解析硬约束)
- Phase 1.4 数据驱动诊断: ✅ 上一轮对话已做 (4 组 HV trajectory 重算)
- Phase 1.5 deliverable: ✅ findings.md + prompts_draft.md 双产出

### Key User Decisions This Session

1. **极简哲学**: "我觉得这一部分最大的问题是写的太多" — 用户多次拒绝
   填充, 倾向最短可行 prompt
2. **保留机器解析硬约束**: 所有 OUTPUT FORMAT 的 JSON 模板 + `{{}}` 转义 +
   ID/status enum 必须保留
3. **Supervisor 加 budget_progress**: 唯一一个 prompt + code 联动改动. F-PROMPT-004
   修复, 让 supervisor 后期能选 no_change
4. **reseed 诱惑 (F-PROMPT-006) 暂不动**: 用户选 9d, 等 v2 phase 4 实测后再
   决定加 lever / 加硬约束 / 加软提示
5. **Scientist insights 消费 (F-PROMPT-003) 暂不动**: 用户选 10b, 不在
   supervisor prompt 强引导

### Next Action

进入 task_plan.md Phase 2 实施:

1. 把 prompts_draft.md 的 6 个 prompt v2 落地到 `efnas/search/prompts.py`
2. 改 `supervisor_agent.py` + `nsga2_search.py` 注入 budget_progress 字段
3. 跑 tests/test_*_agent.py + scripts/smoke_hybrid_v1.py 验证回归
4. 然后进入 Phase 3 (模型升级评估)

待用户决策:
- A. 现在直接进 Phase 2 代码落地 (改 prompts.py 等), 我开干
- B. 先停下来过一遍最终 6 个 v2 草稿全文 (端到端 sanity check), 再落地
- C. 跳过 Phase 2 代码, 直接进 Phase 3 模型评估 (用 v1 prompt + 新模型对照)

### Files modified this session

- 改 `plan/search_hybrid_v2/prompts_draft.md`: 加总览表 + 6 个 prompt v2 草稿
- 改 `plan/search_hybrid_v2/task_plan.md`: Current Phase 推进到 Phase 2
- 改 `plan/search_hybrid_v2/progress.md`: 本会话日志

---

## Session: 2026-05-20 (Phase 2 代码落地)

### Status

- **Status**: 6 个 prompt v2 落地到 `efnas/search/prompts.py`; supervisor budget_progress
  联动改动落地; agent 测试套件 95/95 通过.

### Files modified

- `efnas/search/prompts.py`: 整体重写, 6 个 system prompt 全部换 v2 版本
  - UNIVERSAL_WORLDVIEW: + TASK SEMANTICS / + OPTIMIZATION OBJECTIVE /
    删错误 trade-off 段 / 5→8 lever
  - WARMSTART_AGENT_SYSTEM: 删"关键"段 (指挥均匀采样元凶) + 删 example + 删反向激励
  - SCIENTIST_STAGE_A_SYSTEM: Stage 对比 → 1 行; 操作枚举精简; body 示例删
  - SCIENTIST_STAGE_B1_SYSTEM: 末句压缩; 工程层兜底说明删
  - SCIENTIST_STAGE_B2_SYSTEM: INPUT 的 JSON example 压成字段名清单
  - SUPERVISOR_AGENT_SYSTEM: 加 budget_progress 输入; 删两条哲学/兜底句
- `efnas/search/supervisor_agent.py`:
  - 顶部 docstring 升级到 v2 (8-lever + budget_progress 说明)
  - `invoke_supervisor_agent` 新增 `budget_progress: Optional[Dict]` kwarg
  - user_msg 新增 `# Budget Progress` 区段, 渲染 evals/gen 数值
  - `supervisor_pipeline` 同步加 `budget_progress` kwarg, 透传给 `invoke_supervisor_agent`
- `efnas/baselines/nsga2_search.py`:
  - `_maybe_invoke_supervisor` 计算 `budget_progress` dict (evals_used = (gen+1) * pop_size)
    并传给 `supervisor_pipeline`
  - 注释升级到 v2 说明

### Verification

- prompts.py smoke test: 全部 11 个断言通过 (TASK SEMANTICS / OPTIMIZATION OBJECTIVE /
  8 LEVERS / budget_progress 存在; 旧 5-lever / 均匀采样警告 / trade-off 错误段 不存在)
- `tests.test_supervisor_agent + test_scientist_agent + test_warmstart_agent +
  test_llm_json_retry`: **95/95** passed (0.655s)
- 全套 `unittest discover -s tests`: **343/345** passed (171s); 2 个 pre-existing
  失败, 与本次 v2 改动无关:
  - `test_system_overview_pdf_export`: `reportlab` 模块缺失 (v1 已知 unrelated error)
  - `test_all_variants_build_three_multiscale_outputs`: ablation 模型变体列表
    多了 `edgeflownet_bilinear_gate4x` (代码层注册了 5 个变体, 测试期望 4 个,
    pre-existing 一致性问题, 我的 prompt 改动不可能影响这个)

### Behavioral Notes

- v2 prompt 长度: UNIVERSAL_WORLDVIEW 注入后, warmstart 完整 system_prompt 约 1913 字符
  (v1 约 2050 字符, 净减 ~7%; 实际单个 role 段减幅更大, 因为 UNIVERSAL_WORLDVIEW 几乎持平)
- supervisor 现在能看到 evals_used / evals_total / gen_current / gen_total +
  剩余代数 (gen_total - gen_current - 1)
- 老 supervisor_log.json (v1 4 组数据) 兼容: read_supervisor_log 仍返回旧 entry,
  budget_progress 是新加字段不影响历史 entry 反序列化

### Next Action (待用户决策)

Phase 2 工程层结束, 候选下一步:
- A. 跑一次 dry-run / smoke 验证 budget_progress 真的进入 LLM 调用 (1 LLM call ~$0.01)
- B. 直接进 Phase 3 (模型升级评估): 改 llm_client.py 支持 Anthropic/OpenAI 多 provider
- C. 直接进 Phase 4: 用 v2 prompt + Gemini 3.1 Pro 跑一次 d_v4 完整 800 evals
  (HPC ~3h, 看 v2 prompt 改造是否带得动 HV)
- D. 同时做 B + C: 先 B 改 LLM 层, 再用 Opus 4.7 或 GPT-5 跑 Phase 4 d_v4

---

## Session: 2026-05-20 (Phase 3 LLM 多 provider 落地)

### Status

- **Status**: `llm_client.py` 升到 v2 multi-provider. 25 个新 provider 测试 + 95 个
  旧 agent 测试全部通过, 旧 v1 config (gemini-only) 完全向后兼容.

### Files modified

- `efnas/search/llm_client.py`: 整体重写, v2 multi-provider 路由
  - 新增 `_detect_provider(model_id)`: 按前缀识别 (gemini/anthropic/openai/other)
  - 新增 `_is_gemini_3_or_newer(model_id)`: 识别需要锁 temperature=1.0 的模型
  - 新增 `_build_kwargs(role, model_id, ...)`: 按 provider 过滤参数
    * Gemini 3+: temperature 强制 1.0, 支持 `thinking={"budget_tokens": N}`
    * Anthropic: 保留 temperature, 支持 `thinking`
    * OpenAI: 保留 temperature, 支持 `reasoning_effort` ("low"/"medium"/"high")
    * Other: 透传 temperature
  - 新增 `_track_cost(...)`: 可选成本跟踪 (写 JSONL 日志, 累计 token + 美元)
  - 公共 API `chat()` / `chat_json()` 签名完全兼容 v1

### Files added

- `tests/test_llm_client_v2_providers.py`: 25 个新单元测试
  - 6 个 `_detect_provider` 边界 case
  - 4 个 `_is_gemini_3_or_newer` 检测
  - 3 个 Gemini 3+ kwargs filter (temp 强制 1.0 + thinking 注入 + reasoning_effort 忽略)
  - 3 个 Anthropic kwargs filter
  - 3 个 OpenAI kwargs filter
  - 4 个 common (force_json / override / max_tokens)
  - 2 个 backward-compat (v1 config 不带 v2 字段时正常初始化)

### Config schema 增量 (configs/nsga2_*.yaml)

v2 新增的 llm: 子键 (全部可选, 不指定时无影响):
```yaml
llm:
  models:
    warmstart_agent: "anthropic/claude-opus-4-5-20250929"  # 任意 provider 前缀
  temperature:
    warmstart_agent: 0.7  # Anthropic/OpenAI 接受; Gemini 3+ 自动强制 1.0
  thinking_budget:           # 可选: Anthropic + Gemini 3+ 的 thinking 预算
    warmstart_agent: 8192
  reasoning_effort:          # 可选: OpenAI reasoning model
    supervisor_agent: "high"
  cost_log_path: "outputs/.../metadata/llm_cost_log.jsonl"  # 可选: 成本日志
```

### Verification

- `tests.test_llm_client_v2_providers`: **25/25** passed (0.001s)
- `tests.test_llm_json_retry + test_warmstart_agent + test_scientist_agent +
  test_supervisor_agent`: **95/95** passed (旧测试, 0.666s)

### Next Action (待用户决策)

Phase 3 LLM 层完成. 进 Phase 4 前需要选模型 + 写 config:
- E. 用 **Anthropic Claude Opus 4.7** (最强 reasoning, ~$0.10-0.50/call)
- F. 用 **Anthropic Claude Sonnet 4.5** (性价比好, ~$0.03-0.15/call)
- G. 用 **OpenAI GPT-5 + reasoning_effort=high** (~$0.05-0.20/call)
- H. 先做单次 warmstart 对照 (3 个模型各跑 1 次 LLM + 评估 50 arch ~15min × 3),
   对比 gen 0 HV + rationale 质量后再选

---

## Session: 2026-05-20 (Phase 4 准备 - Opus 4.7 config + slurm)

### Status

- **Status**: 用户决定**全栈使用 Claude Opus 4.7** (已有 ANTHROPIC_API_KEY).
  config + 2 个 slurm 脚本就绪, 待 HPC 提交.

### Model benchmark research (2026-05-20)

针对我们的任务 profile (force_json + reasoning + Python code gen) 调研了:
- Claude Opus 4.7 (2026-04-16 发布): GPQA Diamond 94.2%, SWE-bench 87.6%,
  "precise instruction following", $5/$25 per M tokens
- GPT-5.5 (2026-04-23 发布): Terminal-Bench 82.7%, FrontierMath 51.7%, ~$7.78
- Gemini 3.5 Flash (2026-05-19 发布, 昨天!): MCP-Atlas 83.6% (领先 Opus + GPT),
  4x 更快, $1.50/$9 (最便宜)

用户决策: **全栈 Opus 4.7** (论文金标准 + reasoning 顶配 + 已有 API key).

### Files added

- `configs/nsga2_v4.yaml`: 全栈 anthropic/claude-opus-4-7 配置
  - 5 个 role 全部指向 Opus 4.7
  - temperature 1.0 (匹配 v3 Gemini 强制值, 保证模型为唯一变量)
  - thinking_budget 留作注释 (第一轮跑 baseline Opus, 后续 ablation 加 thinking)
  - request_timeout 240s (Opus 偶有慢调用, 留余量)
  - cost_log_path 留作注释 (用户视需要打开)
- `plan/search_hybrid_v2/run_group_b_v4_opus.slurm`:
  - ablation_group=b (只 warmstart)
  - 1 次 LLM call, 预算 < $0.5
  - 用于纯 warmstart 单点对照 (v1 group_b vs v2 group_b_v4_opus)
- `plan/search_hybrid_v2/run_group_d_v4_opus.slurm`:
  - ablation_group=d (全栈: warmstart + scientist + supervisor)
  - 25-30 LLM calls, 预算 $3-15
  - 用于完整 v2 系统对照 (v1 d_v3 vs v2 d_v4_opus)

### Slurm 关键差异 vs v1 (run_group_d.slurm)

- `GEMINI_API_KEY` → `ANTHROPIC_API_KEY` (key 检查 + 错误提示文案)
- config: `configs/nsga2_v3.yaml` → `configs/nsga2_v4.yaml`
- output_root: `outputs/ablation_phase5` → `outputs/search_hybrid_v2`
- experiment_name: `group_X` → `group_X_v4_opus`
- 其他 (slurm header / GPU 申请 / supernet_experiment_dir) 保持原样

### Next Action (用户操作)

在 HPC 上:
1. 确认 `~/.bashrc` 已 export `ANTHROPIC_API_KEY="sk-ant-..."` 并 re-login
2. (可选) 先提交 b_v4_opus 看 warmstart 质量, 跑完再决定要不要提 d_v4_opus:
   ```bash
   cd $VSC_DATA/test/MCUFlowNet/EdgeFlowNAS
   sbatch plan/search_hybrid_v2/run_group_b_v4_opus.slurm
   # 等 ~3h, 看 outputs/search_hybrid_v2/group_b_v4_opus_*/metadata/warmstart_diagnostics.json
   # 对比 v1 group_b 的同一个文件
   ```
3. 或者直接两个一起提 (Slurm queue 自动排, 跑完 b 后跑 d):
   ```bash
   sbatch plan/search_hybrid_v2/run_group_b_v4_opus.slurm
   sbatch plan/search_hybrid_v2/run_group_d_v4_opus.slurm
   ```
4. 跑完后用 `wrappers/run_phase5_ablation_analysis.py` 加 `--path_*` 覆盖, 对比
   v1 (a/b/c/d_v3) 和 v2 (b_v4_opus / d_v4_opus) 的 HV trajectory + Pareto 端点

## Errors Encountered

(none yet)
