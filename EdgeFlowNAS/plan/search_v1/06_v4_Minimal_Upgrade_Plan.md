# v4 最小化升级计划（第二版）

**创建时间**: 2026-03-10  
**最后更新**: 2026-03-10 (v2.1: 全部 9 项代码改动已实施并通过编译验证)  
**状态**: Implemented  
**核心原则**: 每个改动独立可实施、可验证、可回滚，单个改动不超过 50 行代码。

---

## 背景与动机

06 方案的 **诊断完全正确**，但药方过重：6.5 个建议 × 3-5 层子方案 = 25+ 个改动点，对 ~1200 行核心搜索代码来说是重写级别。更严重的是，建议间存在强耦合（DSL 依赖 search_mode，search_mode 依赖 Pareto Engine，Pareto Engine 依赖 epoch_manifest），无法独立实施。

**v4 的定位**: 空白开局（不 seed v3 数据），全 19683 搜索空间，**验证 Agent 系统自主发现搜索空间结构的能力**。目标不仅是找到最佳子网，更是为论文提供 Agent 系统可行性的实证。

本文档提出 **BUG 修复 2 项 + 设计改进 7 项**，按优先级排列，总代码改动 ~120 行。

---

## BUG-1: Agent D-2 和 D-3 缺少搜索空间语义 (UNIVERSAL_WORLDVIEW)

**优先级**: 最高（必须修复的 Bug）  
**改动量**: 2 行（`prompts.py`）  
**解决问题**: D-2/D-3 不知道 arch_code 9 个位置的物理含义  

### 问题证据

`prompts.py` 中只有 A/B/C/D-1 的 system prompt 以 `UNIVERSAL_WORLDVIEW +` 开头。D-2 和 D-3 的定义直接用 `"""`，完全没有搜索空间知识。

**后果**:
- **D-2** 生成验证脚本时，不知道哪些位对应 backbone depth、哪些对应 head kernel size。如果 D-1 写"Head H2 感受野为 7×7"，D-2 无法知道这对应 `arch_code.split(',')[8]`。
- **D-3** 写 findings 规则时缺少搜索空间背景，合并规则时可能混淆 backbone 位和 head 位的逻辑关系。

### 改什么

```python
# prompts.py
AGENT_D2_SYSTEM = UNIVERSAL_WORLDVIEW + """...  # 原来没有前缀
AGENT_D3_SYSTEM = UNIVERSAL_WORLDVIEW + """...  # 原来没有前缀
```

### 验证标准

D-2 生成的验证脚本中，位置索引与 D-1 假设描述的物理维度一致。

---

## BUG-2: 精准度关键 Agent 的 temperature 统一过高

**优先级**: 最高（与 BUG-1 同时修复）  
**改动量**: ~15 行（`llm_client.py` + `search_v1.yaml`）  
**解决问题**: 6 个 Agent 共用 temperature=0.4，但 C/D-2/D-3 需要更低温度保证精准度  

### 问题分析

| Agent | 职责 | 核心需求 | 当前 temp | 建议 temp |
|-------|------|---------|-----------|-----------|
| A (Strategist) | 分析+规划 | 稳健推理 | 0.4 | 0.35 |
| B (Generator) | 生成候选码 | 格式正确+遵指令 | 0.4 | **0.4（不变）** |
| C (Distiller) | 精确读报告 | 零幻觉 | 0.4 | **0.15** |
| D-1 (Scientist) | 发现模式 | 创造性 | 0.4 | 0.45 |
| D-2 (Coder) | 写正确代码 | 零语法错误 | 0.4 | **0.15** |
| D-3 (Rule Mgr) | 精确合并规则 | 零损失 | 0.4 | **0.25** |

> **关于 B 的 temperature**: B 是编码转换器，所有创造性来自 A 的 `direction_describe`。B 只需要精确遵守 findings 约束 + allocation 指令 + 生成合法格式。重复问题的根因是 P0（缺乏历史输入），不是温度。因此 B 保持 0.4 不变。

### 改什么

1. `search_v1.yaml`:
```yaml
llm:
  temperature:
    agent_a_strategist: 0.35
    agent_b_generator: 0.4
    agent_c_distiller: 0.15
    agent_d_scientist: 0.45
    agent_d_coder: 0.15
    agent_d_rule_manager: 0.25
```

2. `llm_client.py`:
```python
# __init__ 中
if isinstance(llm_cfg.get("temperature"), dict):
    self._temperatures = llm_cfg["temperature"]
    self._default_temperature = 0.4
else:
    self._temperatures = {}
    self._default_temperature = float(llm_cfg.get("temperature", 0.4))

# 新增方法
def _resolve_temperature(self, role: str) -> float:
    config_key = self.ROLE_TO_CONFIG_KEY.get(role)
    if config_key and config_key in self._temperatures:
        return float(self._temperatures[config_key])
    return self._default_temperature
```

### 验证标准

D-2 生成的验证脚本语法正确率提升（无 SyntaxError）。C 的 micro_insight 更精确引用原始数值。

---

## P0: 给 Agent B 喂已评估集合摘要

**优先级**: 最高（第一个功能改动）  
**改动量**: ~15 行（`agents.py`）  
**解决问题**: Agent B 有效率低（当前 ~45%），源于 B 完全不读历史  

### 改什么

在 `invoke_agent_b()` 的 `user_msg` 中追加已评估架构的 coverage 摘要：

```python
# agents.py invoke_agent_b() 中新增
evaluated = file_io.get_evaluated_arch_codes(exp_dir)
coverage_hint = _build_coverage_hint(evaluated)
user_msg += f"\n## 已评估架构摘要（禁止重复生成）:\n{coverage_hint}\n"
```

`_build_coverage_hint(evaluated_set)` 返回 3-5 行结构化文本：
- 已评估总数
- 各维度的值分布（例如 "pos[7]: 0占62%, 1占24%, 2占14%"）
- 最近 20 条已评估的完整 arch_code 列表（直接防撞）

### 不做什么

- 不做 Intent DSL
- 不做 L0-L3 抗幻觉协议
- 不做 Python candidate sampler

### 验证标准

实施后连续 3 个 epoch，Agent B 的有效率（去重后新增 / 总候选）应从 ~45% 提升到 >70%。

### 优点

- 改动最小，立刻见效，完全可逆
- 不影响其他任何 Agent 的输入输出
- 根本原因是"B 看不到历史"，那就让它看

### 缺点

- LLM 仍可能忽略 hint（但概率大幅降低）
- 如果 evaluated_set 超过 1000 条，最近 20 条可能不够——到时候可以改成"未覆盖冷区"

---

## P1: SRAM 从 Agent A 摘要视角移除

**优先级**: 很高（与 P0 可同时实施）  
**改动量**: ~10 行（`agents.py` + `findings.md`）  
**解决问题**: `sram_kb` 恒定 1620.0 KB，在 Agent A 的决策摘要中是噪音  

### 改什么

1. `agents.py` → `_summarize_history()` 中从统计列和展示列移除 `sram_kb`：
   ```python
   for col in ["epe", "fps", "cycles_npu"]:  # 删除 "sram_kb"
   ```
   Top-5 和 Recent-10 展示改为 `["arch_code", "epe", "fps"]`（去掉 sram_kb）

2. `findings.md` 顶部加注释：
   ```markdown
   > **注**: sram_kb 在当前搜索空间中恒定为 1620.0 KB（受尾部 AccumResize2 支配），已从搜索决策摘要中移除。history_archive.csv 保留原始列供人类审计。
   ```

### 不改什么

- **不删除** history_archive.csv 中的 sram_kb 列（人类审计需要）
- **不修改** prompts.py 中的 SRAM 相关文本（Agent C 的 prompt 已经明确说忽略 AccumResize2，Agent A/D-1 看不到 sram 数据自然不会提及）
- **不修改** D-1 的输入（D-1 读全量 CSV 包含 sram_kb，但 findings 注释会引导 D-1 不在此方向提假设）

### 验证标准

实施后 Agent A 的 strategic_reflection 不再出现 SRAM/sram 相关内容。

---

## P1.5: micro_insight 喂入 Agent A 视角

**优先级**: 高（与 P1 同批实施）  
**改动量**: ~3 行（`agents.py`）  
**解决问题**: Agent C 每架构花 1 次 LLM 调用生成的硬件洞察，Agent A 完全看不到  

### 问题证据

- Agent C 在 `eval_worker.py` 为每个架构生成 `micro_insight`（精确的 NPU 瓶颈分析）
- 存入 history_archive.csv
- `_summarize_history()` 只展示 `arch_code, epe, fps, sram_kb`——**无 micro_insight**
- D-1 读全量 CSV 能看到 micro_insight，但 A 看不到

### 改什么

在 `_summarize_history()` 的 Top-5 和 Recent-10 展示中追加 micro_insight 列：

```python
# Top-5 展示
best5 = df_sorted.nsmallest(5, "epe")[["arch_code", "epe", "fps", "micro_insight"]].to_string(index=False)
# Recent-10 展示
recent = df.tail(10)[["arch_code", "epe", "fps", "micro_insight"]].to_string(index=False)
```

### 验证标准

Agent A 的 strategic_reflection 开始引用硬件瓶颈信息（如"大核 Head 导致 NPU 利用率骤降"）辅助规划。

---

## P2: 给 Agent A 加 3 个状态统计量

**优先级**: 高（P0 之后实施）  
**改动量**: ~30 行（`agents.py`）  
**解决问题**: Agent A 自我锚定，看不到策略执行效果  

### 改什么

在 `_summarize_history()` 末尾追加 3 个可计算统计量：

```python
def _summarize_history(df) -> str:
    # ... 现有 code ...
    
    # 新增状态统计
    if "epoch" in df.columns and len(df) > 0:
        max_epoch = int(df["epoch"].max())
        last3 = df[df["epoch"] >= max_epoch - 2]
        effective_last3 = len(last3)
        
        # 简单 Pareto 计数 (EPE 最小化, FPS 最大化)
        pareto_count = _count_pareto_2d(df, "epe", "fps")
        
        lines.append(
            f"\n## 搜索状态统计\n"
            f"- 近3轮有效新样本: {effective_last3}\n"
            f"- 当前 Pareto 前沿点数: {pareto_count}\n"
            f"- 总覆盖率: {len(df)}/19683 ({len(df)/19683*100:.1f}%)\n"
        )
    
    return "\n".join(lines)


def _count_pareto_2d(df, min_col, max_col):
    """计算二维 Pareto 前沿点数 (min_col 越小越好, max_col 越大越好)。"""
    try:
        vals = df[[min_col, max_col]].dropna().astype(float).values
        pareto = []
        for i, (a, b) in enumerate(vals):
            dominated = False
            for j, (c, d) in enumerate(vals):
                if i != j and c <= a and d >= b and (c < a or d > b):
                    dominated = True
                    break
            if not dominated:
                pareto.append(i)
        return len(pareto)
    except:
        return 0
```

### 不做什么

- 不做 `search_mode` 三状态机
- 不做策略执行回执系统
- 不做 `observed_pareto_front.json`
- 不做 `epoch_manifest.json`

### 验证标准

Agent A 的 strategic_reflection 开始引用这些统计量（例如"近3轮仅新增 X 个样本"）。

---

## P3: Scientist D-1 去重主题摘要

**优先级**: 高（与 P2 可同时实施）  
**改动量**: ~20 行（`agents.py`）  
**解决问题**: D-1 反复提出"pos8=2 限速""pos4=2 限速"等已知假设  

### 改什么

在 `invoke_agent_d1()` 中，给 D-1 的 prompt 追加已有假设的主题摘要：

```python
def invoke_agent_d1(llm, exp_dir):
    # ... 现有读 history 的代码 ...
    
    # 新增: 生成已有主题摘要
    existing_assumptions = file_io.read_assumptions(exp_dir)
    findings_text = file_io.read_findings(exp_dir)
    topic_summary = _extract_topic_summary(existing_assumptions, findings_text)
    
    user_msg += (
        f"\n## 已有假设和规则的主题摘要（请勿重复提出类似假设）:\n"
        f"{topic_summary}\n"
        f"请提出与以上主题不同的新假设，例如多维交互条件、局部 Pareto 岛的结构模式等。\n"
    )
```

`_extract_topic_summary()` 实现方式：
- 从 `assumptions.json` 提取每条假设中出现的位置编号 + 关键指标
- 从 `findings.md` 提取已证实的规则 ID 和主题
- 返回紧凑摘要（例如 "已证实: pos7/8=2→FPS<5.5 (A17_30_32), pos0+1=0→EPE>3.9 (A08); 待验证: pos3=2→FPS<5.5 (A28)"）

### 不做什么

- 不做双轨系统（轨道 A / 轨道 B）
- 不做假设的 canonicalization 或 subsumption check

### 验证标准

实施后连续 3 次 D-1 触发，不再产出 "pos7=2" 或 "pos8=2" 类已知主题的假设。

---

## P3.5: D-3 增量写入 + 安全校验

**优先级**: 中高（与 P2/P3 可同时实施）  
**改动量**: ~10 行（`agents.py`）  
**解决问题**: D-3 全量改写 findings.md 有静默丢失旧规则的风险  

### 问题分析

当前 D-3 输出**整个 findings.md 的替换文本**，然后 `file_io.write_findings()` 直接覆写。如果 LLM 输出截断（max_tokens=4096，findings 越长越危险）或 hallucinate 掉一条旧规则，**已证实的真理会被静默删除**。

### 改什么（方案 A: 覆写后校验 + 回滚）

```python
# agents.py invoke_agent_d3() 中
current_findings = file_io.read_findings(exp_dir)
# ... (LLM 调用得到 updated_findings) ...

# 安全校验: 新版本的规则数不应少于旧版本
import re
old_count = len(re.findall(r'\*\*ID\*\*', current_findings))
new_count = len(re.findall(r'\*\*ID\*\*', updated_findings))
if new_count < old_count:
    logger.critical("D-3 输出丢失了规则！old=%d new=%d，回滚到旧版本", old_count, new_count)
    return  # 不覆写
file_io.write_findings(exp_dir, updated_findings)
```

### 改什么（方案 B: 改为增量追加，D-3 只输出新增/合并片段）

修改 D-3 的 prompt，要求只输出**新增或合并的规则片段**（而非整个文件），Engine 层负责追加：
- 优点: 根本消除丢失问题
- 缺点: D-3 失去合并冗余规则的能力，需要额外的合并逻辑

### 建议

先实施 **方案 A**（校验 + 回滚），简单有效。如果 v4 实际运行中 D-3 频繁触发回滚，再切换到方案 B。

---

## P4: Epoch-Level 度量记录（论文数据源）

**优先级**: 中（P0-P3 之后实施，但代码极简单）  
**改动量**: ~20 行（`coordinator.py` + `file_io.py`）  
**解决问题**: 缺少 Agent 系统可行性的量化证据  

### 为什么重要

v4 的核心目标之一是**验证 Agent 系统的自主发现能力**。需要每轮记录的度量：
- effective_rate（B 的有效率随 epoch 的变化）
- findings_count（规则发现速度）
- pareto_count（Pareto 前沿进化）
- total_coverage（空间探索进度）

这些数据直接支撑论文中的 **Agent 学习曲线** 图。

### 改什么

```python
# coordinator.py _run_single_epoch() Phase 7 之后追加
epoch_metrics = {
    "epoch": epoch,
    "candidates_generated": len(candidates),
    "effective_new": len(new_archs),
    "effective_rate": round(len(new_archs) / max(len(candidates), 1), 4),
    "total_evaluated": len(file_io.get_evaluated_arch_codes(self.exp_dir)),
    "findings_count": file_io.count_findings(self.exp_dir),
    "assumptions_count": len(file_io.read_assumptions(self.exp_dir)),
}
file_io.append_epoch_metrics(self.exp_dir, epoch_metrics)
```

`file_io.append_epoch_metrics()`: 追加一行到 `metadata/epoch_metrics.csv`。  
`file_io.count_findings()`: 用 regex 计数 findings.md 中的规则条目数。

### 验证标准

v4 运行后 `epoch_metrics.csv` 有完整记录，可直接 matplotlib 画出学习曲线。

---

## ~~P4-old: 缩小搜索空间~~（已取消）

**状态**: ❌ 已取消  
**原因**: v4 的目标变更为"空白开局验证系统"，不再使用 v3 的 findings 作为 seed。全 19683 空间从头搜索。

---

## 关于语言一致性

**结论**: 维持现状（中文提示 + 英文技术术语/列名）。

经检查 v3 全部输出（strategy_log 46 条反思、findings、assumptions、micro_insight），Agent 输出 **100% 中文，无混用现象**。Gemini 模型在中文 prompt 下表现一致，从 Epoch 0 到 46 质量无退化。

不建议改全英文的理由：
- v3 已实证中文输出质量无问题
- 中文提示对人类审阅更友好
- CSV 列名/技术术语（EPE, FPS, Cycles）保持英文是国际惯例

---

## 暂不实施（保留为 v5 议题）

| 06 建议 | 暂不做的理由 | 触发条件 |
|---------|------------|---------|
| Intent DSL + L0-L3 | P0 足以解决重复问题 | P0 后有效率仍 <70% |
| search_mode 三状态机 | 需要 v4 数据校准阈值 | v4 搜索 30+ epoch 后 |
| 策略执行回执系统 | 有用但工程量中等 | v5 规划时 |
| epoch_manifest + run.log | P4 的 epoch_metrics 已基本覆盖 | 需要更细粒度时 |
| Pareto Engine (独立 JSON) | P2 的 3 个统计量足够 | 需要 hypervolume-guided search 时 |
| 双轨 Scientist | P3 的去重摘要满足需求 | D-1 真正产出无法分类的洞察时 |
| strategy_log 截断 | Gemini 1M context 下暂无溢出问题 | 超过 100 epoch 或出现 lost-in-middle |
| 预算 + 状态触发终止 | 手工 total_epochs 够用 | v5 |
| 全面全英文/全中文 | v3 实证无语言混乱 | 出现语言质量退化时 |

---

## 实施顺序

```
1. BUG-1 + BUG-2 （必须修复，0 风险）
   ↓ 验证: D-2 脚本引用正确位置，C/D-2 输出更精确

2. P0 + P1 + P1.5 （可同时实施，互不依赖）
   ↓ 验证: 跑 3 epoch，确认 B 有效率 >70%，A 看到 micro_insight，A 不再提 SRAM

3. P2 + P3 + P3.5 （可同时实施，互不依赖）
   ↓ 验证: 跑 3 epoch，确认 A 引用统计量，D-1 不再重复主题，D-3 无规则丢失

4. P4 （epoch_metrics 记录）
   ↓ 验证: epoch_metrics.csv 正确记录每轮数据

5. 新建 v4 实验目录，空白 findings/assumptions，开始搜索
```

每步之间可以跑几个 epoch 验证效果，不需要一次性全部上线。

---

## 改动全览

| 优先级 | 项目 | 文件 | 代码量 | 解决问题 |
|--------|------|------|--------|---------|
| BUG | D-2/D-3 加 WORLDVIEW | `prompts.py` | 2 行 | 搜索空间语义缺失 |
| BUG | per-agent temperature | `llm_client.py` + yaml | ~15 行 | C/D-2/D-3 精准度 |
| P0 | B coverage hint | `agents.py` | ~15 行 | B 有效率 45%→70%+ |
| P1 | SRAM 移出 A 摘要 | `agents.py` + `findings.md` | ~10 行 | 决策噪音 |
| P1.5 | micro_insight 喂 A | `agents.py` | ~3 行 | C 的洞察被浪费 |
| P2 | A 的状态统计量 | `agents.py` | ~30 行 | A 自我锚定 |
| P3 | D-1 去重摘要 | `agents.py` | ~20 行 | 假设重复 |
| P3.5 | D-3 覆写安全校验 | `agents.py` | ~10 行 | findings 被篡改风险 |
| P4 | epoch_metrics 记录 | `coordinator.py` + `file_io.py` | ~20 行 | 论文 ablation 数据 |
| **总计** | | | **~125 行** | |

---

## 与 06 方案的对比

| 维度 | 06 方案 | 本方案 |
|------|--------|-------|
| 改动点 | ~25 个 | 9 个（含 2 个 BUG 修复） |
| 代码改动 | ~500+ 行 | ~125 行 |
| 耦合性 | 建议间强耦合 | 每个改动完全独立 |
| 可回滚性 | 难 | 每个可单独回滚 |
| 实施风险 | 高 | 低 |
| 时间到可跑 | 需整体重构+测试 | 边改边跑 |
| 理论上限 | 更高 | 足够好 |
| 系统验证 | 未考虑 | epoch_metrics 直接支撑论文 |

**结论**: v4 = 2 个 BUG 修复 + 7 个精准 diff + 空白开局。06 的重构留给 v5（如果 v4 数据证明需要）。
