"""文件 I/O 工具层：CSV / JSON / Markdown 读写 + 断点恢复 (Rescue) 逻辑。

Phase 1.1 (search_hybrid_v1) 状态：
已移除 assumption/finding 相关的全部 I/O 函数：
- assumptions.json: read/write/append/remove/get_next_id 全部删除
- findings.json + findings.md: registry/render/summarize/upsert/remove/count/parse 全部删除
- write_verification_script: D-2 升格管道删除后无消费者
- _read_legacy_findings_*: 兼容代码不再需要

仍保留的核心职责：
- history_archive.csv 的读写
- epoch_metrics.csv 的读写（schema 暂时保留 findings_count/assumptions_count 列以做向后兼容，
  Phase 1.4 会做 schema 升级）
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
from datetime import datetime
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
    "micro_insight",    # str: Phase 1.1 起始终为空（Agent C 已删除）；Phase 1.2 后改为 layer_profile.json 引用
    "epoch",            # int: 被评估时所属的搜索轮次
    "timestamp",        # str: ISO 格式时间戳
]


def _default_run_state() -> Dict[str, Any]:
    """Return the minimal resumable state surface for one search run.

    Phase 1.1: 移除了 scientist_done / assumptions_evaluated / findings_revalidated
    三个旧 phase 标记，因为对应的 phase 已被删除。
    """
    return {
        "current_epoch": None,
        "phase": "idle",
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

    Phase 1.1: 不再创建 assumptions.json 和 findings.json 占位文件。
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

    _touch_csv(os.path.join(exp_dir, "metadata", "history_archive.csv"))
    _touch_md(os.path.join(exp_dir, "metadata", "search_strategy_log.md"),
              default="# 搜索策略演进日志\n\n")
    _touch_json(os.path.join(exp_dir, "metadata", "run_state.json"),
                default=_default_run_state())
    # Phase 1.5: insights.md 从 templates/insights_template.md 拷贝
    _touch_md(os.path.join(exp_dir, "metadata", "insights.md"),
              default=_load_insights_template())

    logger.info("实验目录已初始化: %s", exp_dir)
    return exp_dir


def _load_insights_template() -> str:
    """读取 efnas/search/templates/insights_template.md 内容. 找不到返回最小骨架."""
    template_path = os.path.join(
        os.path.dirname(__file__), "templates", "insights_template.md",
    )
    if os.path.exists(template_path):
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError:
            logger.warning("无法读取 insights 模板: %s", template_path)
    # Fallback minimum skeleton
    return "# Search Insights\n\n---\n\n"


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
    """将多行评估结果追加到 history_archive.csv。"""
    if not rows:
        return
    path = os.path.join(exp_dir, "metadata", "history_archive.csv")
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            safe_row = {col: row.get(col, "") for col in HISTORY_COLUMNS}
            writer.writerow(safe_row)

    logger.info("已追加 %d 行到 history_archive.csv", len(rows))


# ===================================================================
# tmp_workers/ Map-Reduce 操作
# ===================================================================

def write_worker_result(exp_dir: str, arch_code_str: str, result: Dict[str, Any]) -> str:
    """Worker 线程将单个评估结果写入 tmp_workers/ 下的独立 JSON 文件。"""
    safe_name = arch_code_str.replace(",", "")
    path = os.path.join(exp_dir, "dashboard", "tmp_workers", f"arch_{safe_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path


def collect_and_commit_worker_results(exp_dir: str) -> int:
    """Reduce 阶段：收集 tmp_workers/ 中所有 JSON 并追加到全局 CSV，然后清空。"""
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

    for jf in json_files:
        try:
            os.remove(jf)
        except OSError:
            pass

    logger.info("Reduce 完成: 收集 %d / %d 个临时结果", len(rows), len(json_files))
    return len(rows)


def rescue_orphaned_results(exp_dir: str) -> int:
    """断点恢复 (Rescue)：检查 tmp_workers/ 是否有残存的 JSON，如有则立即入表。"""
    count = collect_and_commit_worker_results(exp_dir)
    if count > 0:
        logger.info("断点恢复: 成功抢救 %d 条遗留评估结果", count)
    return count


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
        # 兼容旧 run_state.json: 旧字段 (scientist_done / assumptions_evaluated /
        # findings_revalidated) 会被 _default_run_state() 过滤掉，不再传播
        state.update({k: v for k, v in data.items() if k in state})
    return state


def write_run_state(exp_dir: str, state: Dict[str, Any]) -> None:
    """全量覆写运行时状态。"""
    merged = _default_run_state()
    merged.update({k: v for k, v in state.items() if k in merged})
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
# Epoch Metrics
# ===================================================================

_LEGACY_EPOCH_METRICS_COLUMNS: List[str] = [
    "epoch", "total_evaluated", "new_evaluated", "duplicates",
    "rule_rejected", "best_epe", "best_fps", "pareto_count",
    "findings_count", "assumptions_count",
    "coverage_pct",
]


def append_epoch_metrics(
    exp_dir: str,
    metrics: Dict[str, Any],
    *,
    columns: Optional[List[str]] = None,
) -> None:
    """追加单条 epoch / generation 指标到 epoch_metrics.csv。

    Phase 1.4: 增加 ``columns`` 关键字参数，让调用方决定要写什么列。
    NSGA-II baseline 在 Phase 1.3 起会传入 search_metrics.GENERATION_METRICS_COLUMNS
    (36 列含 HV / crowding / entropy / gap / stagnation 等)；过渡形态的
    agentic coordinator 不传该参数，沿用 _LEGACY_EPOCH_METRICS_COLUMNS。

    Migration 行为：如果文件已存在但列不一致，把旧行按新列重写，缺失字段填 ""。
    """
    path = os.path.join(exp_dir, "metadata", "epoch_metrics.csv")
    target_columns: List[str] = list(columns) if columns is not None else list(
        _LEGACY_EPOCH_METRICS_COLUMNS
    )

    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_columns = list(reader.fieldnames or [])
            if existing_columns != target_columns:
                old_rows = list(reader)
                with open(path, "w", newline="", encoding="utf-8") as wf:
                    writer = csv.DictWriter(
                        wf, fieldnames=target_columns, extrasaction="ignore",
                    )
                    writer.writeheader()
                    for row in old_rows:
                        writer.writerow({col: row.get(col, "") for col in target_columns})
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=target_columns, extrasaction="ignore")
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
    """Return True only for real search experiment directories.

    Phase 1.1: 移除了对 assumptions.json / findings.json|md 的存在性要求，因为
    新系统不再创建这些文件。仍要求 history_archive.csv、search_strategy_log.md、
    dashboard 子目录存在，作为"agentic 搜索实验目录"的最小特征。
    """
    required_paths = [
        os.path.join(exp_dir, "metadata", "history_archive.csv"),
        os.path.join(exp_dir, "metadata", "search_strategy_log.md"),
        os.path.join(exp_dir, "dashboard", "tmp_workers"),
        os.path.join(exp_dir, "dashboard", "eval_outputs"),
        os.path.join(exp_dir, "scripts"),
    ]
    return all(os.path.exists(path) for path in required_paths)
