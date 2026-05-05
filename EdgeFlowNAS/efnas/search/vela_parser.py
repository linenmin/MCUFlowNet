"""Vela 报告层级数据 Python 解析器 (Phase 1.2, search_hybrid_v1)。

把 NPU 编译器产出的 per-layer CSV 转成结构化 List[Dict]，给 Phase 3 Scientist
做硬件 grounding 时按需查询。这是替代旧 Agent C 每架构 LLM 调用的纯 Python
路径：信息熵从 0（771 条同义复述）提升为可比较的数值向量。

核心设计原则（来自 search_hybrid_v1 Cross-Phase Design Principles）：
- Vela 是 agent 的查询服务，不是驱动信号 → Coordinator 不主动喂 Vela 数据，
  只有 LLM 显式调用 query_vela_for_arch 时才返回
- 数据结构化但不带任何语义判断 → 不在 parser 里产 "瓶颈在 X" 这类自然语言

CSV schema 来自 Vela ethos-u-vela 工具的 *_per-layer.csv 输出：
列顺序固定 15 列，含义见 _PER_LAYER_COLUMN_INDEX。
"""

import csv
import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Vela per-layer CSV 字段索引（位置固定，列名有重复字段如 "Network%"
# 不能用 csv.DictReader）
_PER_LAYER_COLUMN_INDEX: Dict[str, int] = {
    "tflite_op": 0,    # TFLite_operator
    "nng_op": 1,       # NNG Operator
    "sram_bytes": 2,   # SRAM Usage (bytes)
    "peak_pct": 3,     # Peak% (SRAM 峰值占比)
    "cycles": 4,       # Op Cycles
    "cycles_pct": 5,   # Network% (cycles 占网络总数的百分比)
    # 6: NPU cycles - 在我们的 NPU-only 配置下与 cycles 等同
    # 7-10: SRAM/DRAM/Flash AC - 在当前 Grove_Sys_Config 下大多为 0
    "macs": 11,        # MAC Count
    "macs_pct": 12,    # Network% (MACs)
    "util_pct": 13,    # Util% (NPU 利用率)
    "raw_name": 14,    # Name（融合算子可能由 ";" 分隔多段路径）
}

_MIN_REQUIRED_COLUMNS = max(_PER_LAYER_COLUMN_INDEX.values()) + 1

# 架构块短标签匹配模式（顺序决定优先级，更具体的先匹配）
# 用于把 Vela 的层名映射回 11D 搜索空间的某一维所对应的物理块。
# 这是启发式映射；找不到对应时返回 "Other"，Phase 3 Scientist 看到 "Other"
# 自然会忽略而不是误判。
_BLOCK_TAG_PATTERNS = [
    # Heads - "H1Out" / "H2Out" / "H0Out" 必须在 "H1" / "H2" 之前匹配
    ("/H0Out", "H0Out"),
    ("/H1Out", "H1Out"),
    ("/H2Out", "H2Out"),
    ("/H1/", "H1"),
    ("/H2/", "H2"),
    # Encoder/decoder backbones (11D 搜索空间的 EB0/EB1/DB0/DB1 维度)
    ("/EB0", "EB0"),
    ("/EB1", "EB1"),
    ("/DB0", "DB0"),
    ("/DB1", "DB1"),
    # Stems (E0/E1 维度) - 使用融合算子名里的 E0_/E1_ 标记
    ("/E0_", "E0"),
    ("/E1_", "E1"),
    # Down/Up samplers (固定结构，不在 11D 搜索维度里)
    ("Down1", "Down1"),
    ("Down2", "Down2"),
    ("Up1", "Up1"),
    ("Up2", "Up2"),
    # 尾部多尺度聚合（被 Agent C 旧版本反复抱怨的"AccumResize 占 100% SRAM"
    # 噪声层）：保留精确 tag 让 Phase 3 Scientist 自行决定是否过滤
    ("AccumResize1", "AccumResize1"),
    ("AccumResize2", "AccumResize2"),
    ("AccumAdd1", "AccumAdd1"),
    ("AccumAdd2", "AccumAdd2"),
    ("strided_slice", "StridedSliceTail"),
]


def _extract_block_tag(raw_name: str) -> str:
    """从原始 Vela 层名中提取架构块短标签。

    raw_name 可能由 ";" 分隔多段路径（融合算子的多个原始 op）。我们扫描整段
    字符串而不是仅看第一段，这样即使主输出节点不带块名（例如某个 fused 后
    显示为 conv 而不是 EB0），也有机会从依赖路径里识别出来。

    返回 "Other" 表示找不到匹配；返回 "Unknown" 表示输入为空。
    """
    if not raw_name:
        return "Unknown"
    text = raw_name.strip()
    if not text:
        return "Unknown"
    for pattern, tag in _BLOCK_TAG_PATTERNS:
        if pattern in text:
            return tag
    return "Other"


def _parse_float(value: Any, default: float = 0.0) -> float:
    """容错的浮点解析。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value: Any, default: int = 0) -> int:
    """容错的整数解析（兼容 Vela 输出中常见的 1234567.0 形式）。"""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_vela_layer_profile(per_layer_csv_path: str) -> List[Dict[str, Any]]:
    """读取 Vela per-layer CSV，返回结构化层级 profile。

    输入:
        per_layer_csv_path: Vela 工具产出的 *_per-layer.csv 路径

    输出:
        List[Dict]，每个 dict 描述一个层（顺序与 CSV 行序一致），字段:
            - block_tag (str): 架构块短标签
            - tflite_op (str)
            - nng_op (str)
            - sram_bytes (int)
            - peak_pct (float)
            - cycles (int)
            - cycles_pct (float)
            - util_pct (float)
            - macs (int)
            - macs_pct (float)
            - raw_name (str): 原始 Name 字段（保留做 audit）

    错误处理：CSV 不存在 / 无法打开 / 列数不足时返回 []，并 logger.warning。
    单行解析异常时跳过该行，不让单点错误污染整张表。
    """
    if not per_layer_csv_path or not os.path.exists(per_layer_csv_path):
        logger.warning("Vela per-layer CSV 不存在: %s", per_layer_csv_path)
        return []

    rows: List[Dict[str, Any]] = []
    try:
        with open(per_layer_csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return []
            if len(header) < _MIN_REQUIRED_COLUMNS:
                logger.warning(
                    "Vela per-layer CSV 列数 %d < %d，可能格式不符: %s",
                    len(header), _MIN_REQUIRED_COLUMNS, per_layer_csv_path,
                )
                return []

            for row in reader:
                if len(row) < _MIN_REQUIRED_COLUMNS:
                    continue
                try:
                    raw_name = row[_PER_LAYER_COLUMN_INDEX["raw_name"]].strip()
                    rows.append({
                        "block_tag": _extract_block_tag(raw_name),
                        "tflite_op": row[_PER_LAYER_COLUMN_INDEX["tflite_op"]].strip(),
                        "nng_op": row[_PER_LAYER_COLUMN_INDEX["nng_op"]].strip(),
                        "sram_bytes": _parse_int(row[_PER_LAYER_COLUMN_INDEX["sram_bytes"]]),
                        "peak_pct": _parse_float(row[_PER_LAYER_COLUMN_INDEX["peak_pct"]]),
                        "cycles": _parse_int(row[_PER_LAYER_COLUMN_INDEX["cycles"]]),
                        "cycles_pct": _parse_float(row[_PER_LAYER_COLUMN_INDEX["cycles_pct"]]),
                        "util_pct": _parse_float(row[_PER_LAYER_COLUMN_INDEX["util_pct"]]),
                        "macs": _parse_int(row[_PER_LAYER_COLUMN_INDEX["macs"]]),
                        "macs_pct": _parse_float(row[_PER_LAYER_COLUMN_INDEX["macs_pct"]]),
                        "raw_name": raw_name,
                    })
                except (IndexError, AttributeError):
                    logger.warning("跳过格式错误的行: %r", row[:5])
                    continue
    except Exception:
        logger.exception("解析 Vela per-layer CSV 失败: %s", per_layer_csv_path)
        return []

    return rows


def find_per_layer_csv(run_output_dir: str) -> Optional[str]:
    """在评估输出目录中定位 Vela per-layer CSV 文件。

    查找顺序（先具体后宽泛）：
    1. analysis/vela_tmp/**/*per-layer*.csv  (常规 Vela 输出位置)
    2. analysis/vela_tmp/**/*per_layer*.csv  (备用命名)
    3. analysis/**/*per-layer*.csv           (兜底)
    4. analysis/**/*per_layer*.csv           (兜底备用命名)
    5. <run_output_dir>/**/*per-layer*.csv   (legacy 平铺布局)

    Returns:
        匹配到的第一个文件路径；找不到时返回 None。
    """
    if not run_output_dir or not os.path.isdir(run_output_dir):
        return None
    analysis_dir = os.path.join(run_output_dir, "analysis")
    patterns = [
        os.path.join(analysis_dir, "vela_tmp", "**", "*per-layer*.csv"),
        os.path.join(analysis_dir, "vela_tmp", "**", "*per_layer*.csv"),
        os.path.join(analysis_dir, "**", "*per-layer*.csv"),
        os.path.join(analysis_dir, "**", "*per_layer*.csv"),
        os.path.join(run_output_dir, "**", "*per-layer*.csv"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    return None


def write_layer_profile(run_output_dir: str, layer_profile: List[Dict[str, Any]]) -> str:
    """把解析好的层级 profile 持久化到 analysis/layer_profile.json。

    Returns:
        写入的 JSON 文件绝对路径。
    """
    analysis_dir = os.path.join(run_output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    path = os.path.join(analysis_dir, "layer_profile.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(layer_profile, f, ensure_ascii=False, indent=2)
    return path


def parse_and_persist_layer_profile(run_output_dir: str) -> Optional[str]:
    """完整路径：定位 per-layer CSV → 解析 → 写 layer_profile.json。

    Phase 1.2 后 eval_worker 在每次评估成功之后调用这个函数；零 LLM 调用。

    Returns:
        layer_profile.json 的路径；如果找不到 per-layer CSV 或解析为空返回 None。
    """
    csv_path = find_per_layer_csv(run_output_dir)
    if not csv_path:
        logger.info("未找到 Vela per-layer CSV，跳过层级 profile 持久化: %s",
                    run_output_dir)
        return None
    layer_profile = parse_vela_layer_profile(csv_path)
    if not layer_profile:
        logger.warning("Vela per-layer 解析为空: %s", csv_path)
        return None
    output_path = write_layer_profile(run_output_dir, layer_profile)
    logger.info("已持久化层级 profile: %s (%d 层)", output_path, len(layer_profile))
    return output_path


def query_vela_for_arch(exp_dir: str, arch_code: str) -> Optional[List[Dict[str, Any]]]:
    """按 arch_code 查询某子网的层级 profile。

    Phase 3 Scientist agent 在做 hardware grounding 时调用这个函数。
    Coordinator 不主动喂 Vela 数据；只有在 LLM 显式要求时才查询，对应
    "Vela 是 agent 的查询服务，不是驱动信号" 的设计原则。

    Args:
        exp_dir: 实验根目录
        arch_code: 形如 "0,1,2,0,0,1,2,1,0,1,0" 的逗号分隔架构码

    Returns:
        - List[Dict] 同 parse_vela_layer_profile 的输出
        - None 如果该 arch 既无 layer_profile.json 也无原始 per-layer CSV
    """
    if not arch_code:
        return None
    safe_name = arch_code.replace(",", "")
    run_dir = os.path.join(exp_dir, "dashboard", "eval_outputs", f"run_{safe_name}")
    profile_path = os.path.join(run_dir, "analysis", "layer_profile.json")

    if os.path.exists(profile_path):
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            logger.exception("读取 layer_profile.json 失败: %s", profile_path)

    # Fallback: 兼容旧实验目录或者 Vela tflite 已被 prune 但 csv 还在
    csv_path = find_per_layer_csv(run_dir)
    if csv_path:
        return parse_vela_layer_profile(csv_path)
    return None
