# 07 — v5 搜索效能提升计划

> 基于 search_v4_20260310_190031 的 29 epoch 运行数据分析，识别出的搜索效能瓶颈。

## 背景

v4 的 9 项改进已全部验证生效（去重 0%、epoch_metrics 记录、WORLDVIEW 前缀等），
但 29 epoch 后暴露出三个搜索效能瓶颈：

1. Agent B 重复产出率飙升（epoch 28: 19/20 重复 → 仅 1 个新架构）
2. Agent A 策略后期崩溃（copy-paste 同一策略文本）
3. Finding 一旦升格无法降级（错误 finding 永久约束搜索空间）

另有两个已修复的 BUG：
- ✅ BUG-3: `count_findings()` regex 与 findings.md 格式不匹配 → 永远返回 1
- ✅ BUG-4: D3 安全校验 regex 同样不匹配 → 安全检查空转

---

## 改进项

### P1: Agent B 已评估集感知 (减少无效重复生成)

**问题**: Agent B 每轮生成 batch_size=20 个候选，但不知道哪些已被评估。
coordinator Phase 5 过滤了重复（过滤层工作正常），但 Agent B 浪费的产能
导致 new_evaluated 从 20/轮跌到 1/轮。

**方案**: 在 Agent B 的 user message 中注入完整的已评估架构编码列表（模型有 1M 上下文），
同时修改 system prompt 将"禁止重复"升为最高优先级规则。

**实际修改**:
- `agents.py` invoke_agent_b(): 调用 `file_io.get_evaluated_arch_codes()` 获取全量已评估集，
  格式化为排序逗号列表，注入到 user_msg 的 `## 已评估架构列表 (禁止重复生成)` 小节
- `prompts.py` AGENT_B_SYSTEM: 删除 "你不具备反思与学习能力" 等旧规则，
  新增规则 #1: "绝对禁止重复" + 收到「已评估架构列表」不得生成重复编码
- `coordinator.py`: 新增 `self._last_yield_info` 状态变量，Phase 5 后记录有效产出率，
  Phase 3 传给 `invoke_agent_a()` 的 `last_yield_info` 参数
- `agents.py` invoke_agent_a(): 新增 `last_yield_info` 参数，作为 `## 上轮执行反馈` 传入

**Status**: ✅ Complete

---

### P2: Agent A 策略多样性维持 (防止后期策略崩溃)

**问题**: Agent A 从 epoch ~22 开始重复输出几乎相同的策略文本，
反复说"帕累托前沿停滞"而无实质性策略创新。

**方案**: 暂时搁置 — P1 解决后 Agent A 将获得准确的有效产出率反馈
（`_last_yield_info`），有望自行调整策略。如 v5 运行后仍有 collapse，
再实施策略去重或周期性重置。

**Status**: 🔄 Deferred (待 v5 运行验证 P1 效果后决定)

---

### P3: Finding 降级/再验证机制

**问题**: Finding 一旦通过 `confidence >= 0.95 AND total >= 5` 即永久升格，
没有任何方式降级或重新验证。如果验证脚本有 bug、数据量不足导致
假阳性，该 finding 将永久错误地约束搜索空间。

**方案**: 每个 epoch 的 Phase 2 中，先升格猜想 (`_evaluate_pending_assumptions`)，
然后对所有已升格 Finding 重新运行验证脚本 (`_revalidate_findings`)。
置信度 < 0.95（与升格阈值相同）的 Finding 降级回猜想队列。

**实际修改**:
- `file_io.py`: 新增 `parse_findings()` (解析 findings.md 为结构化列表) +
  `remove_finding_by_id()` (按 ID 删除单条 Finding)；`count_findings()` 改为基于 `parse_findings()`
- `coordinator.py`: 新增 `_revalidate_findings()` 方法 — 遍历所有 findings，
  查找对应的 `eval_assumption_{id}.py` 脚本并执行，confidence < threshold 则:
  从 findings.md 移除 + 加回 assumptions.json
- `coordinator.py` `_run_single_epoch()`: Phase 2 后追加 Phase 2b 调用 `_revalidate_findings()`

**参数**: 每 epoch 再验证, 降级阈值 = confidence_threshold (0.95), 无冷却期

**Status**: ✅ Complete

---

## 实施顺序建议

1. **P1** (Agent B 感知) — 直接提升搜索产出效率，效果立竿见影
2. **P2** (Agent A 策略多样性) — 解决后期搜索停滞
3. **P3** (Finding 降级) — 长期正确性保障

---

## 已完成项 (BUG 修复)

- [x] BUG-3: `file_io.count_findings()` regex 改为 `r"^- \*\*ID\*\*:"` (原来用 heading regex)
- [x] BUG-4: `agents.invoke_agent_d3()` 安全校验 regex 同步修改
