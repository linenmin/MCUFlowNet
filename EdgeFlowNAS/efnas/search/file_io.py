"""文件 I/O 工具层：CSV / JSON / Markdown 读写 + 断点恢复 (Rescue) 逻辑。

职责：
- history_archive.csv / epoch_metrics.csv 的读写
- assumptions.json 的增量追加与删除
- findings.json 的注册表读写（兼容旧 findings.md）
- search_strategy_log.md 的人类可读日志追加
- run_state.json 的最小运行时状态管理
- tmp_workers/ 的 Map-Reduce Rescue 机制
- 实验目录的初始化创建
"""

import csv
import glob
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

# history_archive.csv 的列定义（固定顺序）
HISTORY_COLUMNS = [
    "arch_code",        # "0,1,2,0,0,1,2,1,0"
    "epe",              # float: Endpoint Error
    "fps",              # float: inferences_per_second
    "sram_kb",          # float: SRAM 占用 (KB)
    "cycles_npu",       # int: NPU 计算周期
    "macs",             # int: 乘加运算总数
    "micro_insight",    # str: Agent C 蒸馏的硬件简报
    "epoch",            # int: 被评估时所属的搜索轮次
    "timestamp",        # str: ISO 格式时间戳
]


def _default_run_state() -> Dict[str, Any]:
    """Return the minimal resumable state surface for one search run."""
    return {
        "current_epoch": None,
        "phase": "idle",
        "scientist_done": False,
        "assumptions_evaluated": False,
        "findings_revalidated": False,
        "agent_a_result": None,
        "candidates": [],
        "new_archs": [],
        "duplicates": 0,
        "rule_rejected": 0,
        "map_done": False,
        "reduce_done": False,
        "last_yield_info": "",
    }


# ===================================================================
# 实验目录初始化
# ===================================================================

def init_experiment_dir(output_root: str, experiment_name: str) -> str:
    """创建并返回本次实验的根目录路径。

    目录结构:
        {output_root}/{experiment_name}_YYYYMMDD_HHMMSS/
            metadata/
            dashboard/tmp_workers/
            dashboard/eval_outputs/
            scripts/

    Args:
        output_root: 搜索产物根目录 (如 "outputs/search_v1")。
        experiment_name: 实验名称标签。

    Returns:
        实验根目录的绝对路径。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_root, f"{experiment_name}_{timestamp}")

    subdirs = [
        "metadata",
        os.path.join("dashboard", "tmp_workers"),
        os.path.join("dashboard", "eval_outputs"),
        "scripts",
    ]
    for sub in subdirs:
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)

    # 初始化空文件（如果不存在）
    _touch_csv(os.path.join(exp_dir, "metadata", "history_archive.csv"))
    _touch_json(os.path.join(exp_dir, "metadata", "assumptions.json"), default=[])
    _touch_json(os.path.join(exp_dir, "metadata", "findings.json"), default=[])
    _touch_md(os.path.join(exp_dir, "metadata", "search_strategy_log.md"),
              default="# 搜索策略演进日志\n\n")
    _touch_json(os.path.join(exp_dir, "metadata", "run_state.json"), default=_default_run_state())

    logger.info("实验目录已初始化: %s", exp_dir)
    return exp_dir


def find_latest_experiment_dir(output_root: str) -> Optional[str]:
    """查找 output_root 下最新的真实搜索实验目录（用于断点恢复）。"""
    if not os.path.isdir(output_root):
        return None
    candidates = [
        os.path.join(output_root, d)
        for d in os.listdir(output_root)
        if os.path.isdir(os.path.join(output_root, d))
        and _is_valid_search_experiment_dir(os.path.join(output_root, d))
    ]
    if candidates:
        return max(candidates, key=os.path.getmtime)
    return None


# ===================================================================
# history_archive.csv 操作
# ===================================================================

def read_history(exp_dir: str) -> pd.DataFrame:
    """读取全局事实表。返回空 DataFrame 如果文件为空或不存在。"""
    path = os.path.join(exp_dir, "metadata", "history_archive.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    return pd.read_csv(path, dtype={"arch_code": str})


def read_epoch_metrics(exp_dir: str) -> pd.DataFrame:
    """读取 epoch_metrics.csv；若不存在则返回空表。"""
    path = os.path.join(exp_dir, "metadata", "epoch_metrics.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def get_evaluated_arch_codes(exp_dir: str) -> Set[str]:
    """返回已评估过的所有 arch_code 集合（用于去重）。"""
    df = read_history(exp_dir)
    if df.empty:
        return set()
    return set(df["arch_code"].values)


def append_history_rows(exp_dir: str, rows: List[Dict[str, Any]]) -> None:
    """将多行评估结果追加到 history_archive.csv。

    此函数只由主线程在 Reduce 阶段调用，无需加锁。
    """
    if not rows:
        return
    path = os.path.join(exp_dir, "metadata", "history_archive.csv")
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            # 确保所有列都有值
            safe_row = {col: row.get(col, "") for col in HISTORY_COLUMNS}
            writer.writerow(safe_row)

    logger.info("已追加 %d 行到 history_archive.csv", len(rows))


# ===================================================================
# tmp_workers/ Map-Reduce 操作
# ===================================================================

def write_worker_result(exp_dir: str, arch_code_str: str, result: Dict[str, Any]) -> str:
    """Worker 线程将单个评估结果写入 tmp_workers/ 下的独立 JSON 文件。

    Args:
        exp_dir: 实验根目录。
        arch_code_str: 架构编码字符串 (如 "0,1,2,0,0,1,2,1,0")。
        result: 包含评估指标的字典。

    Returns:
        写入的 JSON 文件路径。
    """
    # 用下划线替代逗号，使文件名安全
    safe_name = arch_code_str.replace(",", "")
    path = os.path.join(exp_dir, "dashboard", "tmp_workers", f"arch_{safe_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path


def collect_and_commit_worker_results(exp_dir: str) -> int:
    """Reduce 阶段：收集 tmp_workers/ 中所有 JSON 并追加到全局 CSV，然后清空。

    Returns:
        本次收集并提交的结果数量。
    """
    tmp_dir = os.path.join(exp_dir, "dashboard", "tmp_workers")
    json_files = glob.glob(os.path.join(tmp_dir, "arch_*.json"))

    if not json_files:
        return 0

    rows: List[Dict[str, Any]] = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            rows.append(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("跳过损坏的临时文件 %s: %s", jf, e)

    if rows:
        append_history_rows(exp_dir, rows)

    # 清空 tmp_workers/
    for jf in json_files:
        try:
            os.remove(jf)
        except OSError:
            pass

    logger.info("Reduce 完成: 收集 %d / %d 个临时结果", len(rows), len(json_files))
    return len(rows)


def rescue_orphaned_results(exp_dir: str) -> int:
    """断点恢复 (Rescue)：检查 tmp_workers/ 是否有残存的 JSON，如有则立即入表。

    此函数应在引擎重启后、主循环开始前被第一个调用。

    Returns:
        抢救的结果数量。
    """
    count = collect_and_commit_worker_results(exp_dir)
    if count > 0:
        logger.info("断点恢复: 成功抢救 %d 条遗留评估结果", count)
    return count


# ===================================================================
# assumptions.json 操作
# ===================================================================

def prune_vela_tflite_artifacts(exp_dir: str) -> int:
    """Remove heavy Vela *.tflite artifacts while keeping CSV/TXT analysis files."""
    eval_outputs_dir = os.path.join(exp_dir, "dashboard", "eval_outputs")
    if not os.path.isdir(eval_outputs_dir):
        return 0

    pattern = os.path.join(eval_outputs_dir, "run_*", "analysis", "vela_tmp", "**", "*.tflite")
    removed = 0
    for path in glob.glob(pattern, recursive=True):
        try:
            os.remove(path)
            removed += 1
        except OSError as e:
            logger.warning("删除 Vela tflite 失败: %s (%s)", path, e)
    return removed


def read_assumptions(exp_dir: str) -> List[Dict[str, Any]]:
    """读取猜想队列。"""
    path = os.path.join(exp_dir, "metadata", "assumptions.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def write_assumptions(exp_dir: str, assumptions: List[Dict[str, Any]]) -> None:
    """全量覆写猜想队列。"""
    path = os.path.join(exp_dir, "metadata", "assumptions.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(assumptions, f, ensure_ascii=False, indent=2)


def append_assumptions(exp_dir: str, new_items: List[Dict[str, Any]]) -> None:
    """增量追加猜想到队列末尾。"""
    existing = read_assumptions(exp_dir)
    existing.extend(new_items)
    write_assumptions(exp_dir, existing)


def remove_assumption_by_id(exp_dir: str, assumption_id: str) -> None:
    """删除指定 ID 的猜想（升格为 Finding 后清理用）。"""
    assumptions = read_assumptions(exp_dir)
    filtered = [a for a in assumptions if a.get("id") != assumption_id]
    write_assumptions(exp_dir, filtered)


def get_next_assumption_id(exp_dir: str) -> int:
    """返回下一个可用的猜想序号（从已有 ID 推断）。"""
    assumptions = read_assumptions(exp_dir)
    max_id = 0
    for a in assumptions:
        aid = a.get("id", "")
        if aid.startswith("A") and aid[1:].isdigit():
            max_id = max(max_id, int(aid[1:]))
    for finding in read_findings_registry(exp_dir):
        fid = str(finding.get("id", ""))
        if fid.startswith("A") and fid[1:].isdigit():
            max_id = max(max_id, int(fid[1:]))
    return max_id + 1


# ===================================================================
# findings.json / findings.md 操作
# ===================================================================

def read_findings_registry(exp_dir: str) -> List[Dict[str, Any]]:
    """读取 finding 注册表；若无 JSON，则尝试兼容旧 findings.md。"""
    json_path = os.path.join(exp_dir, "metadata", "findings.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []

    return _read_legacy_findings_as_registry(exp_dir)


def write_findings_registry(exp_dir: str, findings: List[Dict[str, Any]]) -> None:
    """全量覆写 findings.json。"""
    path = os.path.join(exp_dir, "metadata", "findings.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(findings, f, ensure_ascii=False, indent=2)


def upsert_finding(exp_dir: str, finding: Dict[str, Any]) -> None:
    """按 ID 增量写入/替换 finding 注册表条目。"""
    findings = read_findings_registry(exp_dir)
    finding_id = str(finding.get("id", "")).strip()
    updated = []
    replaced = False
    for item in findings:
        if str(item.get("id", "")).strip() == finding_id:
            updated.append(finding)
            replaced = True
        else:
            updated.append(item)
    if not replaced:
        updated.append(finding)
    write_findings_registry(exp_dir, updated)


def render_active_finding_hints(exp_dir: str) -> str:
    """渲染给 Generator 的短约束提示文本。"""
    findings = [f for f in read_findings_registry(exp_dir) if f.get("active", True)]
    if findings:
        lines = []
        for finding in findings:
            hint = str(finding.get("generator_hint") or finding.get("summary") or "").strip()
            if not hint:
                continue
            lines.append(f"- [{finding.get('id', '?')}] {hint}")
        return "\n".join(lines) if lines else "(无 active finding hints)"

    legacy_text = _read_legacy_findings_markdown(exp_dir)
    return legacy_text if legacy_text.strip() else "(无 active finding hints)"


def summarize_active_findings(exp_dir: str) -> str:
    """渲染给 Scientist / human 的 findings 主题摘要。"""
    findings = [f for f in read_findings_registry(exp_dir) if f.get("active", True)]
    if findings:
        lines = []
        for finding in findings:
            title = str(finding.get("title") or finding.get("id") or "?").strip()
            summary = str(finding.get("summary") or finding.get("generator_hint") or "").strip()
            if summary:
                lines.append(f"- [{finding.get('id', '?')}] {title}: {summary}")
            else:
                lines.append(f"- [{finding.get('id', '?')}] {title}")
        return "\n".join(lines) if lines else "(无 active findings)"

    legacy_text = _read_legacy_findings_markdown(exp_dir)
    return legacy_text[:500] if legacy_text.strip() else "(无 active findings)"


def read_findings(exp_dir: str) -> str:
    """兼容旧接口：返回渲染后的 finding 文本。"""
    rendered = render_active_finding_hints(exp_dir)
    if rendered and rendered != "(无 active finding hints)":
        return rendered
    return _read_legacy_findings_markdown(exp_dir)


def write_findings(exp_dir: str, content: str) -> None:
    """兼容旧接口：覆写 legacy findings.md，仅供旧工具链读取。"""
    path = os.path.join(exp_dir, "metadata", "findings.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ===================================================================
# search_strategy_log.md 操作
# ===================================================================

def read_strategy_log(exp_dir: str) -> str:
    """读取战术演进日志。"""
    path = os.path.join(exp_dir, "metadata", "search_strategy_log.md")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def append_strategy_log(exp_dir: str, epoch: int, reflection_text: str) -> None:
    """在战术日志末尾追加一条新的反思记录。"""
    path = os.path.join(exp_dir, "metadata", "search_strategy_log.md")
    entry = (
        f"\n---\n"
        f"## Epoch {epoch} 反思 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n\n"
        f"{reflection_text}\n"
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)


# ===================================================================
# 验证脚本文件操作
# ===================================================================

def write_verification_script(exp_dir: str, filename: str, code: str) -> str:
    """将 Agent D-2 生成的验证脚本保存到 scripts/ 目录。

    Returns:
        脚本的绝对路径。
    """
    path = os.path.join(exp_dir, "scripts", filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    logger.info("验证脚本已保存: %s", path)
    return path


# ===================================================================
# run_state.json 操作
# ===================================================================

def read_run_state(exp_dir: str) -> Dict[str, Any]:
    """读取运行时状态；若不存在则返回默认空态。"""
    path = os.path.join(exp_dir, "metadata", "run_state.json")
    if not os.path.exists(path):
        return _default_run_state()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return _default_run_state()

    state = _default_run_state()
    if isinstance(data, dict):
        state.update(data)
    return state


def write_run_state(exp_dir: str, state: Dict[str, Any]) -> None:
    """全量覆写运行时状态。"""
    merged = _default_run_state()
    merged.update(state)
    path = os.path.join(exp_dir, "metadata", "run_state.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


def update_run_state(exp_dir: str, **updates: Any) -> Dict[str, Any]:
    """增量更新运行时状态并返回更新后的状态。"""
    state = read_run_state(exp_dir)
    state.update(updates)
    write_run_state(exp_dir, state)
    return state


def begin_epoch_state(exp_dir: str, epoch: int, *, last_yield_info: str = "") -> Dict[str, Any]:
    """初始化某个 epoch 的运行状态。"""
    state = _default_run_state()
    state["current_epoch"] = epoch
    state["phase"] = "epoch_started"
    state["last_yield_info"] = last_yield_info
    write_run_state(exp_dir, state)
    return state


def clear_epoch_state(exp_dir: str, *, last_yield_info: str = "") -> None:
    """将运行状态清回 idle，但保留上轮 yield 信息。"""
    state = _default_run_state()
    state["last_yield_info"] = last_yield_info
    write_run_state(exp_dir, state)


# ===================================================================
# Epoch Metrics (P4: 搜索进度可观测性)
# ===================================================================

def parse_findings(exp_dir: str) -> List[Dict[str, str]]:
    """返回 finding 结构化列表（优先 JSON，兼容旧 markdown）。"""
    findings = read_findings_registry(exp_dir)
    return [
        {
            "id": str(item.get("id", "")).strip(),
            "block": json.dumps(item, ensure_ascii=False, indent=2),
        }
        for item in findings
        if str(item.get("id", "")).strip()
    ]


def remove_finding_by_id(exp_dir: str, finding_id: str) -> None:
    """兼容旧接口：将指定 ID 的 Finding 标记为 inactive。"""
    findings = read_findings_registry(exp_dir)
    updated = []
    for finding in findings:
        if finding.get("id") == finding_id:
            patched = dict(finding)
            patched["active"] = False
            updated.append(patched)
        else:
            updated.append(finding)
    write_findings_registry(exp_dir, updated)


def count_findings(exp_dir: str) -> int:
    """计算 findings 注册表中的 active 规则条目数量。"""
    return sum(1 for finding in read_findings_registry(exp_dir) if finding.get("active", True))


def append_epoch_metrics(exp_dir: str, metrics: Dict[str, Any]) -> None:
    """追加单条 epoch 指标到 epoch_metrics.csv。"""
    path = os.path.join(exp_dir, "metadata", "epoch_metrics.csv")
    columns = [
        "epoch", "total_evaluated", "new_evaluated", "duplicates",
        "rule_rejected", "best_epe", "pareto_count", "findings_count", "assumptions_count",
        "coverage_pct",
    ]
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


# ===================================================================
# 内部辅助函数
# ===================================================================

def _touch_csv(path: str) -> None:
    """如果 CSV 文件不存在，创建带表头的空文件。"""
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HISTORY_COLUMNS)


def _touch_json(path: str, default: Any = None) -> None:
    """如果 JSON 文件不存在，用默认值创建。"""
    if os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(default if default is not None else {}, f, ensure_ascii=False, indent=2)


def _touch_md(path: str, default: str = "") -> None:
    """如果 Markdown 文件不存在，用默认内容创建。"""
    if os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(default)


def _read_legacy_findings_markdown(exp_dir: str) -> str:
    """读取旧 findings.md 文本；不存在则返回空串。"""
    path = os.path.join(exp_dir, "metadata", "findings.md")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_legacy_findings_as_registry(exp_dir: str) -> List[Dict[str, Any]]:
    """尽力将旧 findings.md 转成最小 registry 结构。"""
    import re

    content = _read_legacy_findings_markdown(exp_dir)
    if not content.strip():
        return []

    blocks = re.split(r"(?=^- \*\*ID\*\*:|^##\s+A\d+)", content, flags=re.MULTILINE)
    findings: List[Dict[str, Any]] = []
    for block in blocks:
        text = block.strip()
        if not text:
            continue
        m = re.search(r"\b(A\d+)\b", text)
        if not m:
            continue
        finding_id = m.group(1)
        first_line = text.splitlines()[0].strip()
        findings.append(
            {
                "id": finding_id,
                "active": True,
                "title": first_line[:120],
                "summary": text[:500],
                "generator_hint": text[:200],
                "enforcement": "generator_hint_only",
            }
        )
    return findings


def _is_valid_search_experiment_dir(exp_dir: str) -> bool:
    """Return True only for real agent-search experiment directories.

    This guards `--resume` from accidentally picking diagnostic folders such as
    `timing_probe_*` or `rank_consistency_*` that may share the same output root
    but do not contain the full search-run metadata layout.
    """
    required_paths = [
        os.path.join(exp_dir, "metadata", "history_archive.csv"),
        os.path.join(exp_dir, "metadata", "assumptions.json"),
        os.path.join(exp_dir, "metadata", "search_strategy_log.md"),
        os.path.join(exp_dir, "dashboard", "tmp_workers"),
        os.path.join(exp_dir, "dashboard", "eval_outputs"),
        os.path.join(exp_dir, "scripts"),
    ]
    findings_path_ok = (
        os.path.exists(os.path.join(exp_dir, "metadata", "findings.json"))
        or os.path.exists(os.path.join(exp_dir, "metadata", "findings.md"))
    )
    return all(os.path.exists(path) for path in required_paths) and findings_path_ok
