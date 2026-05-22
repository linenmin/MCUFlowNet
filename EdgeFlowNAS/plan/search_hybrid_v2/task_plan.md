# Task Plan: search_hybrid_v2 — v1 全面体检 + 模型升级评估

## Goal

对 `plan/search_hybrid_v1` 的 hybrid LLM-NSGA-II 系统 (5 LLM role + 8-lever
supervisor) 做一次彻底的体检, 找出**前期 LLM 领先在后期被 NSGA-II 追平**这一
现象背后的可控原因, 重新过一遍 5 个 agent 的 prompt / 输入 / 输出契约,
并评估把模型从 Gemini 3.1 Pro 升到更强 reasoning 模型 (Claude Opus 4.7 /
GPT-5 / 其他) 是否能在**相同 800 evals 预算下**把最终 HV 提到 7.524 以上。

落地交付: 一份 `findings.md` (问题清单 + 量化证据), 一份 v2 的 phase 切分,
以及最多 2 个 ablation 重跑（仅 a/d 两组, 不重跑 b/c 节省 HPC 时间）。

## Current Phase

Phase 5 — Paper artifact + 多 seed 验证 (待用户决策启动)
Phase 1-4 全部完成: 6 prompts v2 + LLMClient multi-provider + Opus 4.7 adaptive
thinking + 4 个 v2 run 数据齐. 7-way 对照矩阵 + 完整版本变更链已写到 findings.md
末尾 F-FINAL-* 区段.

### 关键结果速览

| Run | HV | best_epe | Pareto | 备注 |
|---|---:|---:|---:|---|
| v2 b_v4_THINK | **7.5266** ⭐ | 4.0260 | 53 | 单点 HV 冠军, 仅 1 LLM 调用 |
| v2 d_v4_opus | 7.5264 | 4.0237 | 50 | 无 thinking baseline |
| v2 d_v4_THINK | 7.5252 | **4.0237** | **55** ⭐ | HV 抢救能力最强 (+0.614) |
| v1 d_v3 | 7.5244 | **4.0230** | 48 | v1 baseline |
| v1 a (NSGA-only) | 7.5217 | 4.0256 | 51 | random init baseline |

## Phases Overview

| Phase | Name | 核心交付 | 是否动代码 | 是否动 HPC |
|---|---|---|---|---|
| 1 | v1 全面审计 | findings.md 列出所有可控问题 + 量化证据 | ❌ 只读 | ❌ |
| 2 | Prompt / Agent 重写 | 5 个 system prompt + warmstart/scientist/supervisor 接口的 v2 版 | ✅ prompts.py + 注入层 | ❌ |
| 3 | 模型升级评估 | Provider 适配 (Anthropic / OpenAI / Gemini), 单次 warmstart 对照实验 | ✅ llm_client.py + 1 个对照脚本 | ✅ 仅 1 组 warmstart eval |
| 4 | 单点对照重跑 | d_v4 全栈 800 evals × 升级后 prompt + 升级后模型 | ✅ slurm 脚本 + config | ✅ 1 组 × ~3h |
| 5 | 结果分析 + 决策 | 对比 a / d_v3 / d_v4, 出红线判定, 决定是否值得多 seed | ❌ 离线分析 | ❌ |

## Phase 1: v1 全面审计

**核心约束**: 不动任何代码, 只读 prompts.py / llm_client.py / nsga2_search.py
+ 4 组 ablation 的 supervisor_log.json / insights.md / warmstart_diagnostics.json,
把所有发现写到 `findings.md`。这一 phase 是后面所有 phase 的依据。

### 1.1 LLM 调用层审计

- [ ] 读 `efnas/search/llm_client.py`: 确认 LiteLLM 的 `completion` 调用参数
  (temperature, response_format, num_retries, timeout) 是否对每个 provider
  都是合理的
- [ ] 读 `efnas/search/warmstart_agent.py` 和 `scientist_agent.py` 和
  `supervisor_agent.py`: 找出每个 agent 实际传给 LLM 的 system_prompt + user_message
  的完整内容 (含格式化插值)
- [ ] 对比 4 组 ablation 的 supervisor_log.json: 看 5 个 agent 调用是否真的
  按设计在 generation 0 / 2 / 5 / 8 / 11 / 14 触发了
- [ ] 检查 LLM 调用失败时的 fallback 路径是否真的没污染 NSGA-II
- **Status:** pending

### 1.2 Agent 输入信息完整性审计

- [ ] **Warmstart agent**: 当前只看到搜索空间 11D 语义 + 算子细节,
  **不知道** 这是光流估计 / FlyingChairs2 / 172x224 / 是否 distill / teacher 是谁
- [ ] **Scientist agent 三阶段**: 输入是 history_archive (arch_code, EPE, FPS) +
  epoch_metrics, **不知道** EPE 数值的物理意义 (pixel 单位? 多尺度平均?)
  和 supernet 的实际 forward 结构
- [ ] **Supervisor agent**: 输入是 supervisor_state + 8 行 epoch_metrics +
  insights.md + supervisor_log, **不知道** NSGA-II 还剩多少 budget / 这个搜索
  空间总大小 (23,328) / Pareto ceiling 的经验值
- [ ] 列出每个 agent **缺失** 的关键信息, 决定 v2 应该补哪些
- **Status:** pending

### 1.3 Agent 输出契约审计

- [ ] Warmstart 输出: 50 个 arch_code + rationale; rationale 是否真的影响下游决策?
- [ ] Scientist Stage A/B-1/B-2 输出: insights drafts → vela queries → final insights.md;
  insights.md 是否真的被 Supervisor 消费? 如果没被消费, c 组就是无效角色
- [ ] Supervisor 输出: 8 lever actions; 哪些 lever 在 d_v3 真的被用过, 哪些
  从来没被触发?
- [ ] 检查 c 组的 insights.md 演化轨迹 (insights.md.gen{N}.bak), 看 Scientist
  的归纳质量是否随 generation 改善
- **Status:** pending

### 1.4 数据驱动诊断: v1 4 组 trajectory 量化

- [ ] 从 4 组 history_archive.csv 重算 HV trajectory at gen 0/3/5/8/10/13/15
  (已在上轮对话完成, 直接抄进 findings.md)
- [ ] 计算 4 组各自的 **前 100 evals 的 Pareto 端点** (best_epe / best_fps /
  HV at 100), 看 LLM warmstart 的真实价值密度
- [ ] 算 c 组的 Scientist 5 次调用各产生多少个新 insight / retire / under_review
  (从 insights.md.gen{N}.bak diff)
- [ ] 算 d_v3 的 Supervisor 5 次调用各动了哪几个 lever (从 supervisor_log.json),
  做一张 lever 使用频次表
- **Status:** pending

### 1.5 Phase 1 deliverable

- [ ] 把 1.1-1.4 的所有发现写到 `findings.md`, 每条问题打上 (severity, fixable_now,
  needs_model_upgrade) 三个 tag
- [ ] 写一份 Phase 2 的 prompt 重写优先级清单 (按 severity 排序)
- **Status:** pending

## Phase 2: Prompt / Agent 重写

**核心约束**: 改 `efnas/search/prompts.py` 和 agent 的 user_message 注入,
不动 NSGA-II 算子层, 不动 llm_client 的调用契约 (这块 phase 3 才碰)。

### 2.1 UNIVERSAL_WORLDVIEW 注入光流领域先验

- [ ] 补充: 任务是 **optical flow estimation** (输入双帧, 输出每像素 2D 位移场)
- [ ] 补充: 数据集 FlyingChairs2, 训练分辨率 172x224, 评估时 EPE 是
  per-pixel 平均 end-point error (单位 pixel)
- [ ] 补充: supernet 用 EdgeFlowNet teacher (arch=2,1,0,0,0,0,1,1,1,1,1)
  做 distill, 子网共享 backbone 权重, 评估即 forward + EPE 计算
- [ ] 补充: 光流任务的领域先验 — 多尺度上采样头是关键, 大 stem 帮助大位移,
  encoder/decoder backbone 深度影响细节恢复
- [ ] 写完后做一次 sanity check: 长度增加 < 200 词 (避免 prompt 噪声)
- **Status:** pending

### 2.2 Warmstart agent 重写

- [ ] 在 system_prompt 加 **覆盖目标**: 50 个种子里 N 个应该覆盖低 EPE 区
  (深 backbone + 大 stem), M 个覆盖高 FPS 区 (全 0), 剩余覆盖中段
  trade-off (允许 agent 自定 N/M, 但要在 rationale 里说明)
- [ ] 在 user_message 加 baseline reference: "随机初始化 50 个的预期 HV ≈ 5.5,
  你能不能产出 HV ≥ 7.0 的初始种群" (这是 d_v3 已经做到的水平, 给 v2 一个明确门槛)
- [ ] 加 **不许**的硬约束: 不许 50 个全是 (2,1,0,0,...) teacher arch_code
  的微扰 (避免 collapse 到 teacher 邻域)
- **Status:** pending

### 2.3 Scientist agent 三阶段重写

- [ ] Stage A: 在 user_message 注入"已知光流先验"段, 让 Stage A 在归纳时
  能引用先验做假设, 而不是纯数据归纳
- [ ] Stage B-1: 给 vela query 一个示例库 (3 个常用 pattern, e.g.
  "Pareto 上低 EPE 的 5 个", "high-FPS 的 layer profile") — 减少
  agent 自由发挥时漏掉关键查询
- [ ] Stage B-2: 输出契约里加一个**"消费提示"** 字段 — Scientist 显式
  标记哪些 insight 是给 Supervisor 看的 (高优先级), 哪些是给论文写作看的
- [ ] 加备份策略: Stage A 失败时不仅 skip, 还要写一条 "stage_a_failed_at_gen={N}"
  到 supervisor_log, 让 Supervisor 知道 insights.md 没更新
- **Status:** pending

### 2.4 Supervisor agent 重写

- [ ] 在 user_message 注入 **budget 进度**: "已用 evals/总 evals = {n}/800,
  剩余 {800-n}", 让 supervisor 在后期能选择保守
- [ ] 注入 **Pareto ceiling 估计**: 当前 HV / 经验上限 7.524 (来自 v1 4 组
  收敛点) → "你还有多少升空"
- [ ] action space 加一个 **`commit_to_current_pareto`** lever (新增第 9 个):
  把 reseed/local_search/parent_pool_source 全 reset 成 default, 让 NSGA-II
  在后期纯收敛 — 现在 supervisor 在 gen 11/14 还在乱挪 reseed, 这是 d_v3
  Open Question #1 的根因
- [ ] 把 SUPERVISOR_AGENT_SYSTEM 的"all null = no_change"那段保留 (v1
  prompt 已经去掉 "no_change 是最合理输出" 的反向 prior, 但仍要让 no_change
  是合法选项)
- **Status:** pending

### 2.5 单元测试 + smoke test

- [ ] 改完每个 prompt 跑 `tests/test_*_agent.py` 看是否回归 (现有 mock
  会发现 prompt 模板的占位符不一致)
- [ ] 跑 `scripts/smoke_hybrid_v1.py` (HPC 上的 5 LLM role 端到端 smoke)
  确认所有 role 都能解析新 prompt
- **Status:** pending

## Phase 3: 模型升级评估

**核心约束**: 不重跑 b/c, 只为 a/d_v3 升级模型层。LLMClient 改造要让
**provider 可热切换** (一行 config 就能从 Gemini 切到 Anthropic / OpenAI)。

### 3.1 候选模型清单 + 真实可用性核查

- [ ] **Claude Opus 4.7** (anthropic/claude-opus-4-7-20251101 — 待确认实际
  model_id): 顶级 reasoning, 支持 temperature, 支持 thinking budget
- [ ] **Claude Sonnet 4.5** (anthropic/claude-sonnet-4-5): 更便宜, reasoning
  仅次于 Opus, 大量上下文场景性价比高
- [ ] **GPT-5** (openai/gpt-5): 注意 GPT-5.5 截至 2026-05 未发布, 用户问题里
  的 "5.5" 实际是 GPT-5; 支持 reasoning_effort
- [ ] **Gemini 3 Ultra** (gemini/gemini-3-ultra): 同样不接受 temperature, 但
  能力强于 3.1 Pro
- [ ] 用 LiteLLM 试调每个 model_id, 确认 API key 配置 (ANTHROPIC_API_KEY /
  OPENAI_API_KEY / 已有的 GEMINI_API_KEY) + 实际可达
- **Status:** pending

### 3.2 LLMClient 适配多 provider

- [ ] 改 `llm_client.py` 让 `temperature` / `seed` / `reasoning_effort` /
  `thinking_budget` 字段按 provider 自动取舍 (Gemini 3 系列丢弃 temperature,
  Anthropic / OpenAI 保留, Gemini 3 用 thinking_budget 而不是 reasoning_effort)
- [ ] 在 configs/nsga2_v3.yaml 加 `llm.provider_overrides` 段, 让每个 role
  能指定不同 provider (e.g. warmstart 用 Opus, scientist 用 Sonnet, supervisor 用 Gemini)
- [ ] 加 cost tracking: 每次 LLM 调用记录 input/output tokens + 美元成本,
  落到 supervisor_log 旁边的 `llm_cost_log.json`
- [ ] 跑 `tests/test_llm_json_retry.py` + 加 3 个新 case 覆盖 provider 切换
- **Status:** pending

### 3.3 单次 warmstart 对照实验 (廉价)

- [ ] 写 `scripts/warmstart_only_eval.py`: 加载 LLMClient, 调一次 warmstart_agent,
  把 50 个 arch_code 直接喂 supernet 跑评估 (~5 min on 4 GPU)
- [ ] 跑 3-4 次: Gemini 3.1 Pro / Opus 4.7 / GPT-5 / Sonnet 4.5 (如果 token
  预算允许)
- [ ] 对比指标: gen 0 HV / best_epe / best_fps / per-dim entropy /
  rationale 文字质量 (是否引用了光流先验)
- [ ] 把结果写进 findings.md 作为 phase 4 模型选择依据
- **Status:** pending

## Phase 4: 单点对照重跑 (d_v4)

**核心约束**: 只跑 1 组, 不做多 seed (HPC 太贵)。复用 v1 配置只换 prompt
和模型, 比 d_v3 公平对照。

### 4.1 配置准备

- [ ] 复制 `configs/nsga2_v3.yaml` → `configs/nsga2_v4.yaml`, 改 llm 段
  指向 Phase 3 选定的模型
- [ ] 复制 `plan/ablation_phase5/run_group_d.slurm` → `plan/search_hybrid_v2/run_group_d_v4.slurm`
- [ ] 在 slurm 顶部加 API key 检查 (ANTHROPIC_API_KEY 或 OPENAI_API_KEY,
  视 Phase 3 选择)
- [ ] sanity check: 提交前用 `--dry_run` 跑 1 代验证 prompt 注入 + 模型路由都正常
- **Status:** pending

### 4.2 跑 d_v4

- [ ] HPC 上 sbatch run_group_d_v4.slurm
- [ ] 监控 supervisor_log.json 实时输出, 看决策质量是否目测优于 d_v3
- [ ] 跑完后立刻 cp 一份 metadata 到本地 `outputs/search_hybrid_v2/group_d_v4_<ts>/`
- **Status:** pending

## Phase 5: 结果分析 + 决策

### 5.1 4-way 对比

- [ ] 跑 `wrappers/run_phase5_ablation_analysis.py` 加 `--path_d_v4` 覆盖,
  得到 a / d_v3 / d_v4 三组 HV trajectory + Pareto 端点
- [ ] 写一份 v2 的 report.md, 重点:
  - 最终 HV 是否提到 > 7.524 (d_v3 上限)
  - best_epe 是否打破 4.0230 (d_v3 单点最优)
  - budget efficiency (达到 HV=7.5 用了多少 evals) — d_v3 是 450
- **Status:** pending

### 5.2 决策点

- [ ] 如果 d_v4 ≥ d_v3 + 0.005 HV: 写论文叙事用 v2, 标 "model upgrade
  contributes measurable HV"
- [ ] 如果 d_v4 ≈ d_v3 (±0.003): 写论文叙事 "v1 已饱和, 模型差异不显著",
  保留 Gemini 3.1 Pro 节省成本
- [ ] 如果 d_v4 < d_v3: 排查是 prompt 改坏了还是模型选错了, 写一条 Phase 3
  教训进 findings.md
- **Status:** pending

### 5.3 是否值得多 seed

- [ ] 看 d_v4 vs d_v3 差距: 若 > 0.01 HV, 考虑投 2 个额外 seed × 6h = 12h
  HPC 验证方差; 若 < 0.005 HV, 不值得
- [ ] 决策落地到 `findings.md` 末尾
- **Status:** pending

## Key Questions

1. **v1 哪个 agent 角色实际贡献最大?** 数据看是 Warmstart (gen 0 +1.5 HV),
   但 LLM 角色之间的因果链没切开过 — c vs b 的 +0.0014 HV 是否真的来自
   Scientist 还是噪声?
2. **领域先验对结果影响多大?** 这是 v2 的核心赌注。光流先验进 prompt 后,
   warmstart 是否能从 gen 0 HV=7.07 提到 7.20+?
3. **模型升级值不值?** Opus 4.7 / GPT-5 比 Gemini 3.1 Pro 单次调用贵 5-10×,
   总 LLM 成本从 ~$1/run 涨到 ~$5-10/run。HV 提升 0.005 值不值这个钱?
4. **Supervisor 的 reseed 诱惑问题怎么治?** d_v3 gen 8/11/14 还在挪 reseed,
   是给它 `commit_to_current_pareto` lever 还是直接 prompt 里告诉它 "gen 10+
   不许再 reseed"?
5. **是否需要在 v2 加 multi-seed 防御性预算?** v1 单 seed 让所有结论都是
   "可能", 多 seed 才能上论文。但 ROI 不高。

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| 不重跑 b/c | b 和 c 的最终 HV 与 d_v3 在噪声范围内 (±0.002), 重跑无信息增益; b/c 的 prompt 改造留到 v3 |
| 只跑 1 组 d_v4 不跑多 seed | HPC ~3h/run, 4 组 × 3 seeds = 36h 太贵; v2 目标是定性看模型/prompt 升级是否带得动, 不是统计学严格论文级证据 |
| Phase 1 不动代码只读 | 避免在还没诊断完就改动代码引入新 bug; v1 已经经历过 d_v1→d_v2→d_v3 三轮 prompt 迭代, 教训是先诊断再动手 |
| 优先 Anthropic 而非 OpenAI | Opus 4.7 在 reasoning + 长上下文性价比目前最高 (相对 GPT-5 同级别), 且 ANTHROPIC_API_KEY 已经在用户 dev 环境里 |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
|       | 1       |            |

## Notes

- v1 的整套基础设施 (epoch_metrics 37 列 / Vela layer profile / 8 lever
  supervisor / sandbox) **不需要改**, v2 只动 prompt 注入和 LLMClient 的
  provider 适配。这意味着 phase 2-3 工作量比 v1 phase 1 小一个数量级。
- 跑 v2 之前先确认 ANTHROPIC_API_KEY / OPENAI_API_KEY 在 HPC 上可用 — 不要等到
  slurm 排到队才发现 key 没配 (v1 GEMINI_API_KEY 这一坑就遇到过).
- 用户在 v1 phase 5 已经表达过对 supervisor 反复挪 reseed 的不满, v2 phase 2.4
  必须给 supervisor 一个"主动让出"的机制 (commit_to_current_pareto lever).
- Re-read this plan before major decisions
- Log ALL errors - they help avoid repetition
