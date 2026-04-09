"""文件 I/O 工具层：CSV / JSON / Markdown 读写 + 断点恢复 (Rescue) 逻辑。

职责：
- history_archive.csv 的追加写入与去重读取
- assumptions.json 的增量追加与删除
- findings.md 的全量读写
- search_strategy_log.md 的增量追加
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
    _touch_md(os.path.join(exp_dir, "metadata", "findings.md"),
              default="# EdgeFlowNAS Findings (已证实的绝对真理)\n\n> 尚无已证实的发现。\n")
    _touch_md(os.path.join(exp_dir, "metadata", "search_strategy_log.md"),
              default="# 搜索策略演进日志\n\n")

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
    return max_id + 1


# ===================================================================
# findings.md 操作
# ===================================================================

def read_findings(exp_dir: str) -> str:
    """读取已证实的绝对真理文档。"""
    path = os.path.join(exp_dir, "metadata", "findings.md")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_findings(exp_dir: str, content: str) -> None:
    """全量覆写 findings.md（由 Agent D-3 输出驱动）。"""
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
# Epoch Metrics (P4: 搜索进度可观测性)
# ===================================================================

def parse_findings(exp_dir: str) -> List[Dict[str, str]]:
    """解析 findings.md，返回每条 Finding 的结构化列表。

    Returns:
        [{"id": "A02", "block": "- **ID**: A02\n- **规则描述**: ...\n..."}, ...]
    """
    import re
    content = read_findings(exp_dir)
    if not content or not content.strip():
        return []
    # 按 '- **ID**:' 分割，每个块是一条 Finding
    blocks = re.split(r"(?=^- \*\*ID\*\*:)", content, flags=re.MULTILINE)
    results = []
    for block in blocks:
        block = block.strip()
        m = re.match(r"^- \*\*ID\*\*:\s*(\S+)", block)
        if m:
            results.append({"id": m.group(1), "block": block})
    return results


def remove_finding_by_id(exp_dir: str, finding_id: str) -> None:
    """从 findings.md 中删除指定 ID 的 Finding 条目。"""
    findings = parse_findings(exp_dir)
    remaining_blocks = [f["block"] for f in findings if f["id"] != finding_id]
    # 重建文件：保留标题 + 剩余条目
    header = "# EdgeFlowNAS Findings (已证实的绝对真理)\n\n"
    new_content = header + "\n\n".join(remaining_blocks) + "\n" if remaining_blocks else header
    write_findings(exp_dir, new_content)


def count_findings(exp_dir: str) -> int:
    """计算 findings.md 中的规则条目数量 (按 '- **ID**:' 标记计)。"""
    return len(parse_findings(exp_dir))


def append_epoch_metrics(exp_dir: str, metrics: Dict[str, Any]) -> None:
    """追加单条 epoch 指标到 epoch_metrics.csv。"""
    path = os.path.join(exp_dir, "metadata", "epoch_metrics.csv")
    columns = [
        "epoch", "total_evaluated", "new_evaluated", "duplicates",
        "best_epe", "pareto_count", "findings_count", "assumptions_count",
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


def _is_valid_search_experiment_dir(exp_dir: str) -> bool:
    """Return True only for real agent-search experiment directories.

    This guards `--resume` from accidentally picking diagnostic folders such as
    `timing_probe_*` or `rank_consistency_*` that may share the same output root
    but do not contain the full search-run metadata layout.
    """
    required_paths = [
        os.path.join(exp_dir, "metadata", "history_archive.csv"),
        os.path.join(exp_dir, "metadata", "assumptions.json"),
        os.path.join(exp_dir, "metadata", "findings.md"),
        os.path.join(exp_dir, "metadata", "search_strategy_log.md"),
        os.path.join(exp_dir, "dashboard", "tmp_workers"),
        os.path.join(exp_dir, "dashboard", "eval_outputs"),
        os.path.join(exp_dir, "scripts"),
    ]
    return all(os.path.exists(path) for path in required_paths)
