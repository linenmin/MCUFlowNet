# Findings: search_hybrid_v2 v1 体检 + 模型升级评估

每条 finding 带 3 个 tag:
- **severity**: high / medium / low (对最终 HV / 论文叙事的影响)
- **fixable_now**: yes / partial / no (只改 prompt 能不能修)
- **needs_model_upgrade**: yes / no / maybe (是否依赖换更强模型)

---

## F-DATA-001 (high, fixable_now=yes, needs_model_upgrade=no): 4 组 800 evals 全部撞 HV ceiling ≈ 7.524

**证据** (来自上轮对话的 HV 重算, 已落盘到 `outputs/ablation_phase5/` 各组的
history_archive.csv, 参考点 EPE=5.5/FPS=3.0):

| Gen | a (NSGA-II) | b (+Warm) | c (+Sci) | d_v3 (Full) |
|---:|---:|---:|---:|---:|
| 0  | 5.476 | 7.065 | 6.985 | 6.948 |
| 5  | 7.346 | 7.445 | 7.408 | 7.464 |
| 10 | 7.509 | 7.488 | 7.502 | 7.515 |
| 15 | 7.5217 | 7.5245 | 7.5259 | 7.5244 |

- gen 0 LLM warmstart 给 +1.5 HV 大领先, gen 10 起 a 反超 b, gen 15 全部
  在 7.521-7.526 这条带子里, 差距 < 0.005 HV (< 0.07%, 单 seed 方差内)
- best_epe 端点 d_v3 4.0230 是全场最优, 但 HV 上限被 NSGA-II 在 23,328
  搜索空间 × 800 evals (3.4% 覆盖率) 这个组合卡死了

**含义**: 任何 v2 改动如果不动 budget / 搜索空间 / 评估器 (supernet rank
quality), 都不可能把最终 HV 拉过 7.530。v2 的目标应该改成:
- (i) 提升 best_epe 单点 (d_v3 4.0230 → < 4.0)
- (ii) 改善 budget efficiency (达到 HV=7.5 从 d_v3 的 450 evals → < 350?)
- (iii) 改善前期收敛 (gen 5 时把 d 的 HV 拉过 7.46 → 7.48+)

---

## F-PROMPT-001 (high, fixable_now=yes, needs_model_upgrade=no): agent 不知道这是光流估计任务

**证据**: 读 `efnas/search/prompts.py` 的 UNIVERSAL_WORLDVIEW (注入到所有 5 个
role 的 system_prompt 前缀):
- 唯一一处提"光流"的是 `EPE = 光流终端误差 (越低越好)` —— 一行字
- 不提: 任务输入 (双帧 RGB) / 输出 (per-pixel 2D flow field) / 数据集
  (FlyingChairs2 / 172x224) / supernet teacher (EdgeFlowNet) / 是否 distill
- 不提光流领域先验: multi-scale upsample 重要性 / large displacement 需要
  大 receptive field / warping + correlation 在哪里做 (实际上 supernet 没用
  cost volume, 是 plain encoder-decoder, 但 agent 不知道)

**实证**: d_v3 的 warmstart_diagnostics.json rationale 是
> "Provide a diverse initial population covering extremes (all 0s, all maxes)
> and a wide, uniformly distributed sample of the search space."

纯 NAS 教科书思路, 无任何光流领域推理。Scientist insights.md 同理:
I-005 "H1=0(3x3) 占 73.3%" / I-006 "backbone depth 是 Pareto 主杠杆"
都是从 EPE/FPS 数据反向归纳, 不是领域先验。

**修复路径**: Phase 2.1 在 UNIVERSAL_WORLDVIEW 加 6-8 行光流先验段落,
长度 +150 词内。配合 F-PROMPT-005 (注入 teacher arch_code) 一起做。

---

## F-PROMPT-002 (medium, fixable_now=yes, needs_model_upgrade=no): Warmstart agent 无明确覆盖目标

**证据**: 现 WARMSTART_AGENT_SYSTEM 只说 "你怎么平衡多样性与高潜力区域,
完全自决"。`compute_warmstart_diagnostics` 算的 per-dim entropy 落到
diagnostics.json 但 **不门控** —— agent 输出极端不均匀也照样进 NSGA-II。

d_v3 warmstart 的 per-dim entropy 全部接近最大 (ln(3)=1.099 / ln(2)=0.693):
说明 agent 选择了"最大均匀分布"策略, 但这未必是最优。光流领域知道 stem
大 + backbone 深 → 低 EPE, 全 0 → 高 FPS, agent 应该在这两个端点附近多采
样本, 不是均匀洒。

**修复路径**: Phase 2.2 加分布建议: 推荐 10-15 个低 EPE 端 (大数值)、10-15 个
高 FPS 端 (全 0/接近)、剩余 20-30 个均匀中段。允许 agent 调整数字但
要在 rationale 解释。

---

## F-PROMPT-003 (high, fixable_now=partial, needs_model_upgrade=maybe): Scientist 的 insights.md 没有被 Supervisor 真正消费

**证据**:
- 看 `efnas/search/supervisor_agent.py` 输入: `current_insights_md` 字段确实
  传给 LLM
- 看 d_v3 supervisor_log.json 的 rationale: 5 次决策**全部引用 epoch_metrics
  的列名+数值** (gene_entropy_dim_7=0, duplicate_rate=0.74 等), **只有 gen 8**
  prompt 间接提到 "I-004 deep decoder architectures" 这种 insight 名字
- 结果: c 组的 5 次 Scientist 调用 = 15 LLM calls + sandbox 执行, 对 HV
  贡献 = c−b = +0.0014 HV (噪声内)

**含义**: Scientist insights 在 v1 设计上是"自由形式产出物"(F-原则 #4 in v1),
被 Supervisor 选择性引用很正常, 但这意味着 c 组在 ablation 设计上**等价于 b**,
HPC 跑 c 是浪费。

**修复路径**:
- Phase 2.3 让 Stage B-2 加 "consumption_hint" 字段, 显式标记哪些 insight 给
  supervisor 看 (会被注入 supervisor prompt 高优先级位置), 哪些只用于论文叙事
- 这是 partial 修复: 真正解决需要让 NSGA-II 算子能 *机械化消费* insight
  (比如 insight 直接 map 到 frozen_dims), 那是 v3 的事

---

## F-PROMPT-004 (medium, fixable_now=yes, needs_model_upgrade=no): Supervisor 不知道 budget 进度也不知道 ceiling

**证据**: supervisor_agent.py 输入 schema:
- `current_state`: 8 lever 当前数值 ✓
- `recent_metrics`: 最近 8 行 epoch_metrics ✓
- `current_pareto_summary`: Pareto 端点 + 大小 ✓
- `current_insights_md`: 最新 insights ✓
- `supervisor_log`: 历史决策 ✓

但 **没有**:
- 当前 generation 编号 / 总 generation 数 (800/50 = 16 代)
- 已用 evals 数
- Pareto ceiling 经验值 (v1 4 组都在 7.524 收敛, 这是经验上限)

**实证**: d_v3 supervisor gen 14 (倒数第 2 代) 还在调 reseed=20, 完全没考虑
"还剩 100 evals, 再 reseed 进新 random arch 都来不及评估完". 这是
**budget-blind** 决策。

**修复路径**: Phase 2.4 在 supervisor user_message 注入 "budget progress
{n}/800, remaining {800-n} evals, expected next supervisor call at gen {n+3}".
让 supervisor 在后期能主动选择 no_change。

---

## F-PROMPT-005 (low, fixable_now=yes, needs_model_upgrade=no): warmstart 不知道 teacher arch_code

**证据**: supernet_v3_fc2_172x224.yaml 的 `train.distill.teacher_arch_code:
"2 1 0 0 0 0 1 1 1 1 1"` (重 stem + 全浅 backbone + 全部 5x5 heads)。这是
人类经验里光流任务的一个好点。

warmstart agent 完全不知道这个先验, 也就不能 "把 teacher 附近的 5-10 个 1-flip
邻居放进种群" 来引导 NSGA-II 优先探索这附近。

**修复路径**: Phase 2.2 在 warmstart user_message 加一行 "teacher arch_code
is (2,1,0,0,0,0,1,1,1,1,1); you may seed a few neighbors but don't collapse
the whole population to teacher domain".

---

## F-PROMPT-006 (medium, fixable_now=yes, needs_model_upgrade=no): Supervisor reseed 诱惑问题 (d_v3 Open Question #1)

**证据** (来自 v1 progress.md 9/Step 7):
- d_v3 supervisor gen 8/11/14 都选 reseed (30/50/20), 即使 local_search +
  frozen_dims 都用过了
- gen 11 agent 主动关 local_search, rationale 说 "local search 让 algorithm
  重复评估" —— 这是 lever 实现层低效 (1-flip 邻居撞已评估), 而 reseed 是
  唯一"还能动"的滑块
- 后果: gen 11/14 的 reseed 注入随机 arch 几乎被现有 Pareto 支配, 浪费 evals

**修复路径**: 两条互斥方案 (在 Phase 2.4 决策):
- **方案 A**: 加第 9 lever `commit_to_current_pareto: bool` (default false),
  设 true 时强制其他 lever 回 default, supervisor 自己掌控
- **方案 B**: 在 prompt 硬约束 "gen >= 12 不许 reseed", agent 不可违反

倾向 A: 保留 agent 自主权, 给它一个"主动让出"的合法选项, 配合 F-PROMPT-004
budget 进度感知后, agent 应能自己学会 gen 12+ commit.

---

## F-LLM-001 (low, fixable_now=yes, needs_model_upgrade=no): LiteLLM 调用层无明显 bug, 但 retry/timeout 可调优

**证据**: 读 `efnas/search/llm_client.py`:
- `temperature` per-role dict ✓
- `response_format={"type": "json_object"}` ✓ (force JSON)
- `num_retries=3`, `timeout=180s`, `max_tokens=8192` ✓
- `chat_json` 解析失败时有一次 "请重新输出合法 JSON" 修复重试 ✓
- 4 组 supervisor_log.json 共 15 次 supervisor 调用 + 25 次 scientist
  + 4 次 warmstart, 没看到一次 LLM 失败留下的 fallback 痕迹

**潜在小问题**:
- `litellm.suppress_debug_info = True` 全局抑制了 LiteLLM 的 retry warning
  日志, 如果 Gemini 内部 retry 了一次 (没到 num_retries) 不会留痕
- `max_tokens=8192` 对 Scientist Stage B-2 输出完整 insights.md 来说偏紧,
  v1 4 组没出过 truncation 但 v2 加领域先验后 insights.md 可能涨长

**修复路径**: Phase 3.2 加 cost tracking 时顺手把 LiteLLM debug 信息打开一档
+ max_tokens 提到 12288。

---

## F-LLM-002 (high, fixable_now=no, needs_model_upgrade=yes): Gemini 3.1 Pro 不接受 temperature ≠ 1.0

**证据**: configs/nsga2_v3.yaml L23-25 注释明确写明 "Gemini 3.1 系列在 LiteLLM
网关上只接受 temperature=1.0"。这是 **Google 服务端限制**, 不是 LiteLLM 或
本项目代码的 bug。

- Gemini 3 系列 (3 Pro / 3 Flash / 3 Ultra) 是 reasoning model, Google 把
  采样链路封装在内部 thinking 模块, 不再暴露 temperature/top_p/top_k 给客户
- 传 temperature != 1.0 会被 LiteLLM 网关拒 400 或服务端默默改写
- 这意味着想做 "温度扫描 ablation" (warmstart agent 在 T=0.3/0.7/1.0/1.3 下
  种群质量差异) 在 Gemini 3 系列上不可行

**修复路径**:
- 换 Anthropic Claude (支持 temperature) 或 OpenAI GPT (支持 temperature
  + reasoning_effort)
- 若坚持 Gemini, 用 `thinking_budget` 参数控制 reasoning 深度替代 temperature
  扫描

---

## F-MODEL-001 (medium, fixable_now=no, needs_model_upgrade=yes): 候选升级模型清单 + 注意事项

| 候选 | LiteLLM model_id (待验证) | 支持 temperature | 支持 reasoning_effort | 支持 thinking_budget | 估算单次调用成本 |
|---|---|---|---|---|---|
| Gemini 3.1 Pro (v1) | gemini/gemini-3.1-pro-preview | ❌ (locked at 1.0) | ❌ | ✅ | ~$0.01-0.05 |
| Claude Opus 4.7 | anthropic/claude-opus-4-7-20251101 ⚠️待确认 | ✅ | ❌ | ✅ (thinking) | ~$0.10-0.50 |
| Claude Sonnet 4.5 | anthropic/claude-sonnet-4-5 | ✅ | ❌ | ✅ (thinking) | ~$0.03-0.15 |
| GPT-5 | openai/gpt-5 ⚠️待确认 | ✅ | ✅ | ❌ | ~$0.05-0.20 |
| Gemini 3 Ultra | gemini/gemini-3-ultra ⚠️待确认 | ❌ | ❌ | ✅ | ~$0.05-0.20 |

**用户提到的 "GPT-5.5" 不是已发布的 OpenAI 模型** (截至 2026-05-20). 实际可
选最强 OpenAI 是 GPT-5 + reasoning_effort=high。Anthropic 最强是 Claude
Opus 4.7。

**单 run 总 LLM 调用数**: ~25-30 calls (warmstart 1 + scientist 15 + supervisor 5
+ 重试若干). 升级到 Opus 4.7 的总成本估算: ~$3-15/run × 4 ablation runs =
~$12-60。可接受。

**修复路径**: Phase 3.1 实测每个候选 model_id 在用户 API key 下可达, 然后
Phase 3.3 单 warmstart 对照看哪个模型在 gen 0 HV 上最强。

---

## F-DESIGN-001 (low, fixable_now=partial, needs_model_upgrade=no): c 组 (Scientist only) 在 ablation 设计上等价于 b

**证据**:
- c−b 最终 HV = +0.0014, 在单 seed 方差内
- Scientist 的 insights.md 是写给"人 + 论文"的, 不进 NSGA-II 算子
- v1 c 组里 supervisor 没启用, 所以 insights 没有任何机械化下游消费者

**含义**: v1 paper 写作可以这样叙事 "c provides insights as a paper artifact,
but its operational contribution to HV is captured in d (where supervisor
consumes them)". 但 ablation 表里 c 数字基本是 placeholder。

**修复路径**:
- 不再单跑 c 组 (v2 phase 4 只跑 a 和 d_v4)
- 论文里把 c 重新定义为 "Scientist standalone, paper artifact-oriented",
  在 ablation 表注释里说明 (c ≈ b) 是预期结果不是 failure

---

## F-DATA-002 (medium, fixable_now=yes, needs_model_upgrade=no): 4 组前 100 evals 的 HV 价值密度 (待 Phase 1.4 补)

**TODO Phase 1.4**: 重算 a/b/c/d_v3 在 evals=50/100/200/400 时的 HV 和
best_epe, 制作 "budget efficiency vs HV" 曲线。这是 v2 真正能讲的故事:
LLM warmstart 在小 budget 下的价值, 而不是 800 evals 满预算下的收敛点
比较。

---

## F-MODEL-NOTE-001 (info): 模型升级 ≠ 必然更高 HV

**主观判断 (待 Phase 4 数据验证)**:
- d_v3 的 supervisor rationale 已经在 Gemini 3.1 Pro 上达到合理的 NAS
  诊断推理水平 ("rank1_saturation 0.78 indicates severe population collapse")
- 换 Opus 4.7 可能给出 marginally 更好的措辞和更精准的 lever 选择, 但
  Pareto ceiling 没动, 最终 HV 大概率仍在 7.524 附近
- **模型升级真正能动的部分**:
  - warmstart 加入领域先验后的种群质量 (Opus reasoning 能力 + 长 prompt 理解)
  - scientist 在归纳出"非显然 insight"的能力 (e.g. "H2 head 偏 3x3 是因为
    高分辨率细节恢复主要靠 conv kernel 数量而非大小") — 这种推理 Gemini 3.1
    可能给不出
- 所以 v2 phase 5 决策应**优先看 best_epe 和 budget efficiency**, 不是最终 HV

---

## Next Actions (出 Phase 1 时填)

- [ ] (Phase 1.4) 补 F-DATA-002 的预算切片数据
- [ ] (Phase 1.5) 把 Phase 1 全部 findings 转成 Phase 2 重写优先级清单
- [ ] (Phase 2 起) 每完成一项重写, 回到对应 finding 加 "RESOLVED in commit XYZ"
