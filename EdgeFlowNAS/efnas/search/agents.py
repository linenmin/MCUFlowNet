"""Agent 调用封装层。

将 LLM 调用、文件 I/O、Prompt 组装整合为高阶函数，供 Coordinator 直接调度。
每个函数对应一个 Agent 的完整交互周期（读上下文 -> 调 LLM -> 写结果）。
"""

import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

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
        - history_archive.csv (全局事实表)
        - epoch_metrics.csv (搜索健康度)
        - 当前 Pareto 点列表（运行时计算）
        - last_yield_info (上轮有效产出率反馈)

    Returns:
        Agent A 输出的 JSON 字典，包含 strategic_reflection 和 allocation。
    """
    # 组装上下文
    history_df = file_io.read_history(exp_dir)
    history_summary = _summarize_history(history_df)
    epoch_metrics_df = file_io.read_epoch_metrics(exp_dir)
    pareto_section = _summarize_current_pareto(history_df)
    metrics_section = _summarize_epoch_metrics(epoch_metrics_df)
    coverage_section = _build_coverage_hint(exp_dir)

    system_prompt = prompts.AGENT_A_SYSTEM.format(batch_size=batch_size)

    # P1c: 上轮有效产出率反馈
    yield_section = ""
    if last_yield_info:
        yield_section = f"\n## 上轮执行反馈:\n{last_yield_info}\n"

    user_msg = (
        f"# 当前 Epoch: {epoch}\n"
        f"# 本轮预算: {batch_size} 个子网\n\n"
        f"## 历史评估事实摘要 (共 {len(history_df)} 条):\n"
        f"```\n{history_summary}\n```\n\n"
        f"## 当前 Pareto 前沿成员摘要:\n"
        f"```\n{pareto_section}\n```\n\n"
        f"## 搜索空间覆盖率结构:\n"
        f"```\n{coverage_section}\n```\n\n"
        f"## 最近搜索健康度:\n"
        f"```\n{metrics_section}\n```\n\n"
        f"{yield_section}"
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
        - active findings hints (生成约束提示)
        - allocation (来自 Agent A 的名额分配)
        - history (已评估架构编码，用于避免重复)

    Returns:
        逗号分隔的架构编码字符串列表 (如 ["2,1,2,1,2,1,0,1,0,1,0", ...])。
    """
    findings = file_io.render_active_finding_hints(exp_dir)
    coverage_hint = _build_coverage_hint(exp_dir)

    # P1: 给 B 全量已评估列表，避免重复生成
    evaluated_set = file_io.get_evaluated_arch_codes(exp_dir)
    evaluated_list_str = ", ".join(sorted(evaluated_set)) if evaluated_set else "(尚无)"

    user_msg = (
        f"## 本轮名额分配 (来自战略规划局):\n"
        f"```json\n{json.dumps(allocation, ensure_ascii=False, indent=2)}\n```\n\n"
        f"## Active Finding Hints:\n{findings}\n\n"
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
        - active findings 摘要（用于去重与避免重复主题）

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
    history_for_scientist = history_df.drop(columns=["micro_insight"], errors="ignore")
    csv_text = history_for_scientist.to_csv(index=False)
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

    filename = result.get("target_filename", f"rule_{assumption.get('id', 'unknown')}.py")
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
    """调用 Agent D-3 将已证实猜想升格写入 findings.json。

    Reads:
        - 当前 findings registry
        - 被升格的 assumption 及其置信度

    Side effects:
        - 更新 findings.json
        - 从 assumptions.json 中删除该猜想
    """
    current_findings = file_io.read_findings_registry(exp_dir)
    rule_script = os.path.join("scripts", f"rule_{assumption.get('id')}.py")

    user_msg = (
        f"## 被证实的新真理:\n"
        f"- ID: {assumption.get('id')}\n"
        f"- 描述: {assumption.get('description')}\n"
        f"- 置信度: {confidence:.4f}\n\n"
        f"## 当前 findings registry:\n"
        f"```json\n{json.dumps(current_findings, ensure_ascii=False, indent=2)}\n```\n\n"
        f"## 对应规则脚本:\n- {rule_script}\n\n"
        f"请输出新 finding 的 registry 条目。\n"
    )

    result = llm.chat_json(
        role="agent_d3",
        system_prompt=prompts.AGENT_D3_SYSTEM,
        user_message=user_msg,
    )
    finding_payload = dict(result.get("finding") or {})
    if not finding_payload:
        logger.error("[Agent D-3] 未返回 finding 条目，放弃升格: %s", assumption.get("id"))
        return

    finding_payload["id"] = assumption.get("id")
    finding_payload["confidence"] = round(float(confidence), 4)
    finding_payload["script"] = finding_payload.get("script") or rule_script
    finding_payload["active"] = bool(finding_payload.get("active", True))
    finding_payload["support"] = finding_payload.get("support", None)
    file_io.upsert_finding(exp_dir, finding_payload)
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
    commands = [
        [sys.executable, script_path, "--mode", "verify", "--data_csv", data_csv_path],
        [sys.executable, script_path, "--data_csv", data_csv_path],
    ]
    try:
        result = None
        for cmd in commands:
            trial = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if trial.returncode == 0:
                result = trial
                break
            result = trial
        if result is None or result.returncode != 0:
            logger.error("验证脚本执行失败: %s\nstderr: %s", script_path, result.stderr[-500:] if result else "")
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


def execute_candidate_check_script(
    script_path: str,
    arch_code: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """执行规则脚本的单候选检查模式。"""
    context_json = json.dumps(context or {}, ensure_ascii=False)
    cmd = [
        sys.executable,
        script_path,
        "--mode",
        "check",
        "--arch_code",
        arch_code,
        "--context_json",
        context_json,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning("规则检查脚本执行失败: %s\nstderr: %s", script_path, result.stderr[-500:])
            return None

        stdout = result.stdout.strip()
        for line in reversed(stdout.split("\n")):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return None
    except subprocess.TimeoutExpired:
        logger.warning("规则检查脚本超时: %s", script_path)
        return None
    except Exception:
        logger.exception("规则检查脚本异常: %s", script_path)
        return None


def filter_candidates_by_findings(
    exp_dir: str,
    arch_codes: List[str],
    *,
    context: Optional[Dict[str, Any]] = None,
) -> tuple[List[str], int]:
    """用 active hard-filter findings 对候选做最终引擎裁决。"""
    findings = [f for f in file_io.read_findings_registry(exp_dir) if f.get("active", True)]
    if not findings:
        return arch_codes, 0

    kept: List[str] = []
    rejected = 0
    for arch in arch_codes:
        blocked = False
        for finding in findings:
            if finding.get("enforcement") != "hard_filter":
                continue
            script_rel = str(finding.get("script") or "").strip()
            if not script_rel:
                continue
            script_path = os.path.join(exp_dir, script_rel.replace("/", os.sep))
            if not os.path.exists(script_path):
                continue
            result = execute_candidate_check_script(script_path, arch, context=context)
            if result and result.get("reject", False):
                blocked = True
                rejected += 1
                logger.info(
                    "[Engine] 候选被 active finding 拒绝: arch=%s, rule=%s, reason=%s",
                    arch,
                    finding.get("id", "?"),
                    result.get("reason", ""),
                )
                break
        if not blocked:
            kept.append(arch)
    return kept, rejected


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
    findings = file_io.summarize_active_findings(exp_dir)

    lines = []
    if assumptions:
        lines.append("### 现有猜想 (assumptions.json):")
        for a in assumptions:
            lines.append(f"  - [{a.get('id', '?')}] {a.get('description', '')[:120]}")
    else:
        lines.append("### 现有猜想: (无)")

    if findings and findings.strip():
        lines.append(f"### 已确认规则 (findings registry 摘要):\n{findings[:500]}")
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

    recent_cols = [c for c in ["arch_code", "epe", "fps"] if c in df.columns]
    recent = df.tail(10)[recent_cols].to_string(index=False)
    lines.append(f"\n最近 10 条评估:\n{recent}")

    return "\n".join(lines)


def _summarize_current_pareto(df, limit: int = 20) -> str:
    """生成当前 Pareto 成员列表摘要，显式展示 EPE 端与 FPS 端。"""
    if df.empty or "epe" not in df.columns or "fps" not in df.columns:
        return "(尚无 Pareto 点)"

    try:
        work = df.copy()
        work["epe"] = work["epe"].astype(float)
        work["fps"] = work["fps"].astype(float)
    except (ValueError, TypeError):
        return "(Pareto 数据解析失败)"

    values = work[["epe", "fps"]].to_numpy()
    pareto_idx: List[int] = []
    for i in range(len(values)):
        dominated = False
        for j in range(len(values)):
            if i == j:
                continue
            if (
                values[j, 0] <= values[i, 0]
                and values[j, 1] >= values[i, 1]
                and (values[j, 0] < values[i, 0] or values[j, 1] > values[i, 1])
            ):
                dominated = True
                break
        if not dominated:
            pareto_idx.append(i)

    if not pareto_idx:
        return "(尚无 Pareto 点)"

    cols = [c for c in ["arch_code", "epe", "fps", "epoch"] if c in work.columns]
    pareto_df = work.iloc[pareto_idx][cols]
    half = max(1, limit // 2)
    low_epe = pareto_df.sort_values(by=["epe", "fps"], ascending=[True, False]).head(half).to_string(index=False)
    high_fps = pareto_df.sort_values(by=["fps", "epe"], ascending=[False, True]).head(half).to_string(index=False)
    return f"Lowest-EPE end:\n{low_epe}\n\nHighest-FPS end:\n{high_fps}"


def _summarize_epoch_metrics(df) -> str:
    """生成最近若干轮搜索健康度摘要。"""
    if df is None or getattr(df, "empty", True):
        return "(尚无 epoch 级健康度数据)"

    cols = [
        c for c in [
            "epoch",
            "new_evaluated",
            "duplicates",
            "rule_rejected",
            "pareto_count",
            "best_epe",
            "best_fps",
        ]
        if c in df.columns
    ]
    if not cols:
        return "(epoch metrics 缺少关键列)"
    recent = df.tail(8)[cols]
    return recent.to_string(index=False)


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
