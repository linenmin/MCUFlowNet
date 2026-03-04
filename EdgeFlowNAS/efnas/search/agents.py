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
) -> Dict[str, Any]:
    """调用 Agent A 生成本轮搜索策略与名额分配。

    读取:
        - history_archive.csv (全局数据板)
        - search_strategy_log.md (历史战术日记)
        - assumptions.json (科学猜想簿)
        - findings.md (绝对真理碑)

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

    user_msg = (
        f"# 当前 Epoch: {epoch}\n"
        f"# 本轮预算: {batch_size} 个子网\n\n"
        f"## 历史评估数据摘要 (共 {len(history_df)} 条):\n"
        f"```\n{history_summary}\n```\n\n"
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

    Returns:
        逗号分隔的架构编码字符串列表 (如 ["0,1,2,0,0,1,2,1,0", ...])。
    """
    findings = file_io.read_findings(exp_dir)

    user_msg = (
        f"## 本轮名额分配 (来自战略规划局):\n"
        f"```json\n{json.dumps(allocation, ensure_ascii=False, indent=2)}\n```\n\n"
        f"## 绝对真理碑 (findings.md) — 你必须遵守:\n{findings}\n\n"
        f"请生成 **恰好 {batch_size} 个** 合法的 9 维架构编码。\n"
    )

    result = llm.chat_json(role="agent_b", system_prompt=prompts.AGENT_B_SYSTEM, user_message=user_msg)

    candidates = result.get("generated_candidates", [])
    # 基础合法性校验
    valid = []
    for c in candidates:
        parts = [p.strip() for p in c.split(",")]
        if len(parts) == 9 and all(p in ("0", "1", "2") for p in parts):
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

    Returns:
        新猜想列表 [{"id": "A01", "description": "..."}, ...]
    """
    history_df = file_io.read_history(exp_dir)
    if history_df.empty:
        logger.info("[Agent D-1] 历史数据为空，跳过猜想提出")
        return []

    next_id = file_io.get_next_assumption_id(exp_dir)

    system_prompt = prompts.AGENT_D1_SYSTEM.format(next_id=next_id)

    # 将完整 CSV 文本化喂入（历史数据量在万级以下，可以全量输入）
    csv_text = history_df.to_csv(index=False)
    user_msg = (
        f"## 全量历史评估数据 (共 {len(history_df)} 条):\n"
        f"```csv\n{csv_text}\n```\n\n"
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

def _summarize_history(df) -> str:
    """生成历史数据的简洁统计摘要（避免全量 CSV 塞爆 prompt）。"""
    if df.empty:
        return "(尚无历史数据)"

    lines = [
        f"总评估数量: {len(df)}",
    ]

    for col in ["epe", "fps", "sram_kb", "cycles_npu"]:
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

    # 展示最近 10 条和最佳 5 条 (按 EPE 排序)
    if "epe" in df.columns:
        try:
            df_sorted = df.copy()
            df_sorted["epe"] = df_sorted["epe"].astype(float)
            best5 = df_sorted.nsmallest(5, "epe")[["arch_code", "epe", "fps", "sram_kb"]].to_string(index=False)
            lines.append(f"\nTop-5 最低 EPE:\n{best5}")
        except (ValueError, TypeError):
            pass

    recent = df.tail(10)[["arch_code", "epe", "fps", "sram_kb"]].to_string(index=False)
    lines.append(f"\n最近 10 条评估:\n{recent}")

    return "\n".join(lines)
