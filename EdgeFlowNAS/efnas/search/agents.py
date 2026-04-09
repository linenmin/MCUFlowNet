"""Agent 调用封装层。

将 LLM 调用、文件 I/O、Prompt 组装整合为高阶函数，供 Coordinator 直接调度。
每个函数对应一个 Agent 的完整交互周期（读上下文 -> 调 LLM -> 写结果）。
"""

import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from efnas.search import file_io
from efnas.search.llm_client import LLMClient
from efnas.search import prompts

logger = logging.getLogger(__name__)


# ===================================================================
# Agent A: 架构规划局 (Strategist)
# ===================================================================

def invoke_agent_a(
    llm: LLMClient,
    exp_dir: str,
    epoch: int,
    batch_size: int,
    last_yield_info: str = "",
) -> Dict[str, Any]:
    """调用 Agent A 生成本轮搜索策略与名额分配。

    读取:
        - history_archive.csv (全局数据板)
        - search_strategy_log.md (历史战术日记)
        - assumptions.json (科学猜想簿)
        - findings.md (绝对真理碑)
        - last_yield_info (上轮有效产出率反馈)

    Returns:
        Agent A 输出的 JSON 字典，包含 strategic_reflection 和 allocation。
    """
    # 组装上下文
    history_df = file_io.read_history(exp_dir)
    history_summary = _summarize_history(history_df)
    strategy_log = file_io.read_strategy_log(exp_dir)
    assumptions = file_io.read_assumptions(exp_dir)
    findings = file_io.read_findings(exp_dir)

    system_prompt = prompts.AGENT_A_SYSTEM.format(batch_size=batch_size)

    # P1c: 上轮有效产出率反馈
    yield_section = ""
    if last_yield_info:
        yield_section = f"\n## 上轮执行反馈:\n{last_yield_info}\n"

    user_msg = (
        f"# 当前 Epoch: {epoch}\n"
        f"# 本轮预算: {batch_size} 个子网\n\n"
        f"## 历史评估数据摘要 (共 {len(history_df)} 条):\n"
        f"```\n{history_summary}\n```\n\n"
        f"{yield_section}"
        f"## 过往战术日志:\n{strategy_log}\n\n"
        f"## 当前猜想簿:\n```json\n{json.dumps(assumptions, ensure_ascii=False, indent=2)}\n```\n\n"
        f"## 绝对真理碑 (findings.md):\n{findings}\n"
    )

    result = llm.chat_json(role="agent_a", system_prompt=system_prompt, user_message=user_msg)

    # 将反思写入战术日志
    reflection = result.get("strategic_reflection", "")
    if reflection:
        file_io.append_strategy_log(exp_dir, epoch, reflection)

    logger.info("[Agent A] 策略生成完毕: allocation=%s", json.dumps(result.get("allocation", {}), ensure_ascii=False))
    return result


# ===================================================================
# Agent B: 编码机器 (Generator)
# ===================================================================

def invoke_agent_b(
    llm: LLMClient,
    exp_dir: str,
    allocation: Dict[str, Any],
    batch_size: int,
) -> List[str]:
    """调用 Agent B 根据策略分配生成候选架构编码列表。

    读取:
        - findings.md (绝对真理碑，作为硬约束)
        - allocation (来自 Agent A 的名额分配)
        - history (已评估架构编码，用于避免重复)

    Returns:
        逗号分隔的架构编码字符串列表 (如 ["2,1,2,1,2,1,0,1,0,1,0", ...])。
    """
    findings = file_io.read_findings(exp_dir)
    coverage_hint = _build_coverage_hint(exp_dir)

    # P1: 给 B 全量已评估列表，避免重复生成
    evaluated_set = file_io.get_evaluated_arch_codes(exp_dir)
    evaluated_list_str = ", ".join(sorted(evaluated_set)) if evaluated_set else "(尚无)"

    user_msg = (
        f"## 本轮名额分配 (来自战略规划局):\n"
        f"```json\n{json.dumps(allocation, ensure_ascii=False, indent=2)}\n```\n\n"
        f"## 绝对真理碑 (findings.md) — 你必须遵守:\n{findings}\n\n"
        f"## 搜索覆盖率统计:\n{coverage_hint}\n\n"
        f"## 已评估架构列表 (禁止重复生成):\n{evaluated_list_str}\n\n"
        f"请生成 **恰好 {batch_size} 个** 合法的、**不在上述已评估列表中的** 11 维架构编码。\n"
    )

    result = llm.chat_json(role="agent_b", system_prompt=prompts.AGENT_B_SYSTEM, user_message=user_msg)

    candidates = result.get("generated_candidates", [])
    # 基础合法性校验
    valid = []
    for c in candidates:
        parts = [p.strip() for p in c.split(",")]
        if (
            len(parts) == 11
            and all(p in ("0", "1", "2") for p in parts[:6])
            and all(p in ("0", "1") for p in parts[6:])
        ):
            valid.append(",".join(parts))
        else:
            logger.warning("[Agent B] 丢弃非法编码: %s", c)

    logger.info("[Agent B] 生成 %d 个候选 (合法 %d 个)", len(candidates), len(valid))
    return valid


# ===================================================================
# Agent D-1: 首席科学家 — 猜想提出
# ===================================================================

def invoke_agent_d1(
    llm: LLMClient,
    exp_dir: str,
) -> List[Dict[str, Any]]:
    """调用 Agent D-1 分析历史数据并提出新的科学猜想。

    读取:
        - history_archive.csv (全量数据)
        - assumptions.json (现有猜想，用于去重)
        - findings.md (已确认规则，用于去重)

    Returns:
        新猜想列表 [{"id": "A01", "description": "..."}, ...]
    """
    history_df = file_io.read_history(exp_dir)
    if history_df.empty:
        logger.info("[Agent D-1] 历史数据为空，跳过猜想提出")
        return []

    next_id = file_io.get_next_assumption_id(exp_dir)
    topic_summary = _extract_topic_summary(exp_dir)

    system_prompt = prompts.AGENT_D1_SYSTEM.format(next_id=next_id)

    # 将完整 CSV 文本化喂入（历史数据量在万级以下，可以全量输入）
    csv_text = history_df.to_csv(index=False)
    user_msg = (
        f"## 全量历史评估数据 (共 {len(history_df)} 条):\n"
        f"```csv\n{csv_text}\n```\n\n"
        f"## 已有猜想和规则 (请勿重复这些方向):\n{topic_summary}\n\n"
        f"请基于以上数据提出 1-2 个新的科学猜想。下一个可用 ID 从 A{next_id:02d} 开始。\n"
    )

    result = llm.chat_json(role="agent_d1", system_prompt=system_prompt, user_message=user_msg)

    new_assumptions = result.get("assumptions", [])
    if new_assumptions:
        file_io.append_assumptions(exp_dir, new_assumptions)
        logger.info("[Agent D-1] 提出 %d 个新猜想: %s",
                     len(new_assumptions),
                     [a.get("id") for a in new_assumptions])
    return new_assumptions


# ===================================================================
# Agent D-2: 数据科研助手 — 验证代码生成
# ===================================================================

def invoke_agent_d2(
    llm: LLMClient,
    exp_dir: str,
    assumption: Dict[str, Any],
) -> Optional[str]:
    """调用 Agent D-2 为单个猜想生成 Python 验证脚本。

    读取:
        - assumption 的描述文本
        - history_archive.csv 的列名（仅表头）

    Returns:
        生成的脚本文件路径，或 None（如果生成失败）。
    """
    history_df = file_io.read_history(exp_dir)
    csv_columns = ", ".join(history_df.columns.tolist()) if not history_df.empty else ", ".join(file_io.HISTORY_COLUMNS)

    system_prompt = prompts.AGENT_D2_SYSTEM.format(csv_columns=csv_columns)

    user_msg = (
        f"## 待验证的科学猜想:\n"
        f"- ID: {assumption.get('id', 'unknown')}\n"
        f"- 描述: {assumption.get('description', '')}\n\n"
        f"请为此猜想编写独立的 Python 验证脚本。\n"
    )

    result = llm.chat_json(role="agent_d2", system_prompt=system_prompt, user_message=user_msg)

    filename = result.get("target_filename", f"eval_assumption_{assumption.get('id', 'unknown')}.py")
    code = result.get("python_code", "")

    if not code:
        logger.error("[Agent D-2] 未生成代码内容")
        return None

    script_path = file_io.write_verification_script(exp_dir, filename, code)
    logger.info("[Agent D-2] 验证脚本已生成: %s", script_path)
    return script_path


# ===================================================================
# Agent D-3: 规则管理者 — Findings 升格
# ===================================================================

def invoke_agent_d3(
    llm: LLMClient,
    exp_dir: str,
    assumption: Dict[str, Any],
    confidence: float,
) -> None:
    """调用 Agent D-3 将已证实猜想升格写入 findings.md。

    Reads:
        - 当前 findings.md 全文
        - 被升格的 assumption 及其置信度

    Side effects:
        - 更新 findings.md
        - 从 assumptions.json 中删除该猜想
    """
    current_findings = file_io.read_findings(exp_dir)

    # 安全基线: 记录更新前的规则数量 (按 '- **ID**:' 标记计)
    import re
    _FINDING_PATTERN = re.compile(r"^- \*\*ID\*\*:", re.MULTILINE)
    rule_count_before = len(_FINDING_PATTERN.findall(current_findings))

    user_msg = (
        f"## 被证实的新真理:\n"
        f"- ID: {assumption.get('id')}\n"
        f"- 描述: {assumption.get('description')}\n"
        f"- 置信度: {confidence:.4f}\n\n"
        f"## 当前 findings.md 全文:\n"
        f"```markdown\n{current_findings}\n```\n\n"
        f"请输出更新后的完整 findings.md 内容。\n"
    )

    updated_findings = llm.chat(
        role="agent_d3",
        system_prompt=prompts.AGENT_D3_SYSTEM,
        user_message=user_msg,
        force_json=False,  # D-3 输出纯 Markdown
    )

    # 安全校验: 确保没有丢失已有规则
    rule_count_after = len(_FINDING_PATTERN.findall(updated_findings))
    if rule_count_before > 0 and rule_count_after < rule_count_before:
        logger.warning(
            "[Agent D-3] 安全回滚! 规则数从 %d 降至 %d，可能丢失已有规则。保留原文并追加。",
            rule_count_before, rule_count_after,
        )
        # 回退策略: 保留原文，仅追加新规则摘要
        updated_findings = (
            f"{current_findings}\n\n"
            f"## [{assumption.get('id')}] {assumption.get('description', '')}\n"
            f"- 置信度: {confidence:.4f}\n"
            f"- (自动追加 — D3 全文重写触发安全回滚)\n"
        )

    file_io.write_findings(exp_dir, updated_findings)
    file_io.remove_assumption_by_id(exp_dir, assumption.get("id", ""))

    logger.info("[Agent D-3] Finding 升格完成: %s (confidence=%.4f)", assumption.get("id"), confidence)


# ===================================================================
# Engine 层: 猜想验证执行器 (非 Agent，纯 Python)
# ===================================================================

def execute_verification_script(
    script_path: str,
    data_csv_path: str,
) -> Optional[Dict[str, Any]]:
    """执行 Agent D-2 生成的验证脚本并解析置信度结果。

    Args:
        script_path: 验证脚本路径。
        data_csv_path: history_archive.csv 路径。

    Returns:
        解析后的结果字典 {"total_triggered": int, "expected_met": int, "confidence": float}
        或 None（如果执行失败）。
    """
    cmd = [sys.executable, script_path, "--data_csv", data_csv_path]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            logger.error("验证脚本执行失败: %s\nstderr: %s", script_path, result.stderr[-500:])
            return None

        # 从 stdout 解析 JSON
        stdout = result.stdout.strip()
        # 尝试找到最后一行合法 JSON
        for line in reversed(stdout.split("\n")):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

        logger.error("验证脚本未输出有效 JSON: %s\nstdout: %s", script_path, stdout[-500:])
        return None

    except subprocess.TimeoutExpired:
        logger.error("验证脚本超时: %s", script_path)
        return None
    except Exception:
        logger.exception("验证脚本异常: %s", script_path)
        return None


# ===================================================================
# 内部辅助函数
# ===================================================================

def _build_coverage_hint(exp_dir: str) -> str:
    """构建搜索空间覆盖率摘要，帮助 Agent B 避免重复生成。"""
    evaluated = file_io.get_evaluated_arch_codes(exp_dir)
    total_space = (3 ** 6) * (2 ** 5)  # 23328
    coverage_pct = len(evaluated) / total_space * 100
    lines = [
        f"已评估架构数: {len(evaluated)} / {total_space} ({coverage_pct:.1f}%)",
    ]
    # 每维度的值分布
    if evaluated:
        from collections import Counter
        dim_counters = [Counter() for _ in range(11)]
        for code in evaluated:
            parts = code.split(",")
            if len(parts) == 11:
                for i, v in enumerate(parts):
                    dim_counters[i][v.strip()] += 1
        for i, counter in enumerate(dim_counters):
            dist = ", ".join(f"{v}:{counter.get(v, 0)}" for v in ["0", "1", "2"])
            lines.append(f"  dim[{i}] 分布: {dist}")
    return "\n".join(lines)


def _extract_topic_summary(exp_dir: str) -> str:
    """提取现有猜想和 findings 的主题摘要，帮助 D1 避免重复。"""
    assumptions = file_io.read_assumptions(exp_dir)
    findings = file_io.read_findings(exp_dir)

    lines = []
    if assumptions:
        lines.append("### 现有猜想 (assumptions.json):")
        for a in assumptions:
            lines.append(f"  - [{a.get('id', '?')}] {a.get('description', '')[:120]}")
    else:
        lines.append("### 现有猜想: (无)")

    if findings and findings.strip():
        lines.append(f"### 已确认规则 (findings.md, 前500字):\n{findings[:500]}")
    else:
        lines.append("### 已确认规则: (无)")

    return "\n".join(lines)


def _summarize_history(df) -> str:
    """生成历史数据的简洁统计摘要（避免全量 CSV 塞爆 prompt）。"""
    if df.empty:
        return "(尚无历史数据)"

    lines = [
        f"总评估数量: {len(df)}",
    ]

    # --- 基础统计 (sram_kb 恒定 1620，排除) ---
    for col in ["epe", "fps", "cycles_npu"]:
        if col in df.columns:
            numeric = df[col].dropna()
            if len(numeric) > 0:
                try:
                    numeric = numeric.astype(float)
                    lines.append(
                        f"{col}: min={numeric.min():.4f}, max={numeric.max():.4f}, "
                        f"mean={numeric.mean():.4f}, std={numeric.std():.4f}"
                    )
                except (ValueError, TypeError):
                    pass

    # --- 操作态统计 (帮 Agent A 感知搜索进度) ---
    total_space = (3 ** 6) * (2 ** 5)
    coverage_pct = len(df) / total_space * 100
    lines.append(f"\n搜索覆盖率: {len(df)}/{total_space} ({coverage_pct:.1f}%)")

    if "epe" in df.columns and "fps" in df.columns:
        try:
            pareto_count = _count_pareto_2d(df["epe"].astype(float), df["fps"].astype(float), minimize_first=True)
            lines.append(f"2D Pareto前沿数量 (EPE↓, FPS↑): {pareto_count}")
        except (ValueError, TypeError):
            pass

    if "epoch" in df.columns:
        try:
            epochs = df["epoch"].astype(int)
            max_epoch = epochs.max()
            if max_epoch >= 3:
                last3 = df[epochs >= max_epoch - 2]
                if "epe" in df.columns:
                    global_best = df["epe"].astype(float).min()
                    last3_best = last3["epe"].astype(float).min()
                    improved = last3_best < global_best * 1.001  # 0.1% tolerance
                    lines.append(f"近3轮是否有改进: {'✓ 有' if improved else '✗ 无 (可能停滞)'}")
        except (ValueError, TypeError):
            pass

    # --- 展示最佳 5 条 (按 EPE 排序)，附带 micro_insight ---
    display_cols = ["arch_code", "epe", "fps"]
    if "micro_insight" in df.columns:
        display_cols.append("micro_insight")

    if "epe" in df.columns:
        try:
            df_sorted = df.copy()
            df_sorted["epe"] = df_sorted["epe"].astype(float)
            best5_cols = [c for c in display_cols if c in df_sorted.columns]
            best5 = df_sorted.nsmallest(5, "epe")[best5_cols].to_string(index=False)
            lines.append(f"\nTop-5 最低 EPE:\n{best5}")
        except (ValueError, TypeError):
            pass

    recent_cols = [c for c in display_cols if c in df.columns]
    recent = df.tail(10)[recent_cols].to_string(index=False)
    lines.append(f"\n最近 10 条评估:\n{recent}")

    return "\n".join(lines)


def _count_pareto_2d(obj1, obj2, minimize_first: bool = True) -> int:
    """计算 2D Pareto 前沿点数量。

    Args:
        obj1: EPE (越小越好 if minimize_first=True)
        obj2: FPS (越大越好)
    """
    import numpy as np
    arr1 = np.array(obj1, dtype=float)
    arr2 = np.array(obj2, dtype=float)
    n = len(arr1)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # j dominates i if j is no worse on both objectives and strictly better on at least one
            if minimize_first:
                j_dom = (arr1[j] <= arr1[i] and arr2[j] >= arr2[i] and
                         (arr1[j] < arr1[i] or arr2[j] > arr2[i]))
            else:
                j_dom = (arr1[j] >= arr1[i] and arr2[j] >= arr2[i] and
                         (arr1[j] > arr1[i] or arr2[j] > arr2[i]))
            if j_dom:
                is_pareto[i] = False
                break
    return int(is_pareto.sum())
