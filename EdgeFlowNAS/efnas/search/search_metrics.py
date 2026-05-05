"""Phase 1.3 (search_hybrid_v1): NSGA-II 搜索健康度监控指标计算。

提供 8 大类指标的纯 Python 计算函数 + 一个聚合函数
`compute_full_generation_metrics()`，让 NSGA-II runner 在每代结束后一行
调用就能拿到全套指标 dict 直接喂给 file_io.append_epoch_metrics 写 CSV。

8 大类指标（详见 plan/search_hybrid_v1/task_plan.md Phase 1.3）：
  (a) Hypervolume (HV / 超体积)
  (b) HV 改进率 (hv_improvement_rate_3gen)
  (c) 平均拥挤距离 (mean_crowding_distance, 去除 ±inf 边界)
  (d) 每维基因熵 (gene_entropy_dim_0..10) -- 当前种群的每维 Shannon 熵
  (e) 停滞代数三个独立计数 (stagnation_best_epe / best_fps / hv)
  (f) 最大 Pareto gap (largest_gap_fps_low/high, largest_gap_epe_low/high)
  (g) 重复率趋势 (duplicate_rate, duplicate_rate_3gen_avg)
  (h) 第一前沿饱和度 (rank1_saturation = 当前种群第一前沿大小 / population_size)

设计原则：
- 纯函数 + dict 输出，无 I/O，便于单元测试
- HV 参考点 (ref_epe=5.5, ref_fps=3.0) 是默认值，可由配置覆盖
- 所有指标都给出明确的边界 case 行为（空前沿、单点前沿、首代、缺数据）
"""

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# 默认 HV 参考点
# ---------------------------------------------------------------------------
# 选取依据：基于 search_v2_refactor_run2 实测数据，最差子网约 (EPE=5.0, FPS=3.5)
# 外推一格作为 ref，保证整个观测到的 Pareto 前沿都在 ref 内侧。
DEFAULT_HV_REF_EPE: float = 5.5
DEFAULT_HV_REF_FPS: float = 3.0


# ---------------------------------------------------------------------------
# generation_metrics CSV schema (按写入顺序)
# ---------------------------------------------------------------------------
# Phase 1.4: 在原 11 列 legacy schema 基础上新增 21 列。
# 保留 'epoch' 列名（同时表示 epoch 或 generation，看搜索模式）以减少向后兼容
# 风险；保留 findings_count/assumptions_count（永远 0）作为 legacy 兼容。
GENERATION_METRICS_COLUMNS: List[str] = [
    # 基础计数
    "epoch",
    "total_evaluated",
    "new_evaluated",
    "duplicates",
    "duplicate_rate",
    "duplicate_rate_3gen_avg",
    "rule_rejected",
    # Pareto 前沿端点 + 尺寸
    "best_epe",
    "best_fps",
    "pareto_count",
    "rank1_saturation",
    # Hypervolume
    "hv",
    "hv_ref_epe",
    "hv_ref_fps",
    "hv_improvement_rate_3gen",
    # 多样性 (前沿层面 + 种群层面)
    "mean_crowding_distance",
    "gene_entropy_dim_0", "gene_entropy_dim_1", "gene_entropy_dim_2",
    "gene_entropy_dim_3", "gene_entropy_dim_4", "gene_entropy_dim_5",
    "gene_entropy_dim_6", "gene_entropy_dim_7", "gene_entropy_dim_8",
    "gene_entropy_dim_9", "gene_entropy_dim_10",
    # 停滞计数
    "stagnation_best_epe",
    "stagnation_best_fps",
    "stagnation_hv",
    # Pareto 前沿几何
    "largest_gap_fps_low",
    "largest_gap_fps_high",
    "largest_gap_epe_low",
    "largest_gap_epe_high",
    # Legacy compat (永远 0)
    "findings_count",
    "assumptions_count",
    # 覆盖率
    "coverage_pct",
]


# ---------------------------------------------------------------------------
# (a) Hypervolume
# ---------------------------------------------------------------------------

def hypervolume_2d(
    pareto_points: Sequence[Tuple[float, float]],
    ref_epe: float = DEFAULT_HV_REF_EPE,
    ref_fps: float = DEFAULT_HV_REF_FPS,
) -> float:
    """计算 2D Pareto 前沿在 (EPE↓, FPS↑) 下的 Hypervolume。

    形式化定义：HV = 前沿和参考点 (ref_epe, ref_fps) 围出的"楼梯"区域面积。
    任何 epe ≥ ref_epe 或 fps ≤ ref_fps 的点都被视为劣解，不计入。

    Args:
        pareto_points: 形如 [(epe, fps), ...] 的列表（不必预先排序）
        ref_epe: 参考 EPE 上界
        ref_fps: 参考 FPS 下界

    Returns:
        非负 float。空前沿或所有点都被参考点拒绝时返回 0.0。
    """
    valid = [(e, f) for (e, f) in pareto_points if e < ref_epe and f > ref_fps]
    if not valid:
        return 0.0
    valid.sort(key=lambda p: p[1])  # FPS ascending

    hv = 0.0
    prev_fps = ref_fps
    for epe, fps in valid:
        if fps <= prev_fps:
            # 同 FPS 多点：因为 sort by FPS 后，相邻同 FPS 的点 EPE 可能不同；
            # 真正属于 Pareto 前沿的话只可能保留 EPE 最低的；这里 skip 重复 FPS。
            continue
        hv += (fps - prev_fps) * (ref_epe - epe)
        prev_fps = fps
    return hv


# ---------------------------------------------------------------------------
# (d) Shannon 熵 + 每维基因熵
# ---------------------------------------------------------------------------

def shannon_entropy(values: Iterable[Any]) -> float:
    """Shannon entropy (natural log base) 对类别分布。

    H = -sum(p * ln(p))。
    Returns 0.0 for empty / single-value sequences.
    """
    counts: Dict[Any, int] = {}
    total = 0
    for v in values:
        counts[v] = counts.get(v, 0) + 1
        total += 1
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log(p)
    return h


def per_dim_gene_entropy(arch_codes: Sequence[str], num_dims: int = 11) -> List[float]:
    """对架构码列表的每一维计算 Shannon 熵。

    Args:
        arch_codes: 形如 ["0,1,2,0,...", ...] 的列表（当前 50 个种群个体）
        num_dims: 期望维度数（11 for V2/V3 search space）

    Returns:
        长度 num_dims 的 float 列表，每元素是该维度取值分布的 Shannon 熵。
        空列表或全部解析失败时返回全 0。
    """
    if not arch_codes:
        return [0.0] * num_dims

    parsed: List[List[str]] = []
    for code in arch_codes:
        parts = [p.strip() for p in str(code).split(",")]
        if len(parts) == num_dims:
            parsed.append(parts)
    if not parsed:
        return [0.0] * num_dims

    return [shannon_entropy([row[d] for row in parsed]) for d in range(num_dims)]


# ---------------------------------------------------------------------------
# (c) 平均拥挤距离
# ---------------------------------------------------------------------------

def mean_crowding_distance_excluding_inf(
    pareto_points: Sequence[Tuple[float, float]],
) -> float:
    """计算 Pareto 前沿的平均拥挤距离（NSGA-II 标准定义），排除 ±inf 边界点。

    每个点的拥挤距离 = sum 各目标维度上左右邻居的归一化间距。
    两端点天然为 +inf；本函数返回 INTERIOR 点的平均距离。

    Returns:
        - 0.0 当前沿点数 ≤ 2（全是边界）
        - 否则 INTERIOR 点的算术平均
    """
    if len(pareto_points) <= 2:
        return 0.0

    n = len(pareto_points)
    distances: List[float] = [0.0] * n
    # 转成双 minimization: (epe, -fps)
    objectives = [(p[0], -p[1]) for p in pareto_points]

    for obj_idx in range(2):
        sorted_idx = sorted(range(n), key=lambda i: objectives[i][obj_idx])
        distances[sorted_idx[0]] = float("inf")
        distances[sorted_idx[-1]] = float("inf")
        min_v = objectives[sorted_idx[0]][obj_idx]
        max_v = objectives[sorted_idx[-1]][obj_idx]
        if max_v == min_v:
            continue
        for k in range(1, n - 1):
            i = sorted_idx[k]
            if distances[i] == float("inf"):
                continue
            left = objectives[sorted_idx[k - 1]][obj_idx]
            right = objectives[sorted_idx[k + 1]][obj_idx]
            distances[i] += (right - left) / (max_v - min_v)

    interior = [d for d in distances if d != float("inf")]
    if not interior:
        return 0.0
    return sum(interior) / len(interior)


# ---------------------------------------------------------------------------
# (f) 最大 Pareto gap
# ---------------------------------------------------------------------------

def largest_pareto_gap(
    pareto_points: Sequence[Tuple[float, float]],
) -> Dict[str, Any]:
    """找 Pareto 前沿按 FPS 排序后的最大相邻间距。

    Returns:
        dict with keys:
            - fps_low, fps_high: 间距两端点的 FPS（low < high）
            - epe_low, epe_high: 对应的 EPE
            - fps_span: high - low
        前沿点数 < 2 时所有字段为空字符串。
    """
    if len(pareto_points) < 2:
        return {
            "fps_low": "",
            "fps_high": "",
            "epe_low": "",
            "epe_high": "",
            "fps_span": "",
        }

    sorted_pts = sorted(pareto_points, key=lambda p: p[1])
    max_gap = -1.0
    pair = (sorted_pts[0], sorted_pts[1])
    for i in range(len(sorted_pts) - 1):
        gap = sorted_pts[i + 1][1] - sorted_pts[i][1]
        if gap > max_gap:
            max_gap = gap
            pair = (sorted_pts[i], sorted_pts[i + 1])
    return {
        "fps_low": float(pair[0][1]),
        "fps_high": float(pair[1][1]),
        "epe_low": float(pair[0][0]),
        "epe_high": float(pair[1][0]),
        "fps_span": float(max_gap),
    }


# ---------------------------------------------------------------------------
# (e) 停滞代数
# ---------------------------------------------------------------------------

def stagnation_count(values: Sequence[float], direction: str = "decrease") -> int:
    """计算"自上次该指标朝指定方向真正改进以来过了几代"。

    Args:
        values: 各代的值序列（按时间升序，最后一个是当前代）
        direction:
            - 'decrease': 更小算改进（适合 EPE）
            - 'increase': 更大算改进（适合 FPS, HV）

    Returns:
        非负整数。当前代是首次出现该最优值时返回 0；当前代未改进
        running_best 时返回距离上次改进的代数。
    """
    if not values or len(values) == 1:
        return 0

    if direction == "decrease":
        # 走一遍数组，记录每个位置的 running_best；找最后一次"严格小于之前
        # running_best"的位置
        running_best = values[0]
        last_improvement_idx = 0
        for i in range(1, len(values)):
            if values[i] < running_best:
                running_best = values[i]
                last_improvement_idx = i
            elif values[i] < running_best:  # 这个分支永远走不到，逻辑安全
                pass
        return len(values) - 1 - last_improvement_idx
    elif direction == "increase":
        running_best = values[0]
        last_improvement_idx = 0
        for i in range(1, len(values)):
            if values[i] > running_best:
                running_best = values[i]
                last_improvement_idx = i
        return len(values) - 1 - last_improvement_idx
    else:
        raise ValueError(f"direction must be 'decrease' or 'increase', got {direction}")


# ---------------------------------------------------------------------------
# Helper: 2D Pareto 前沿提取
# ---------------------------------------------------------------------------

def _compute_pareto_front_2d(
    points: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """返回非支配点子集 (EPE 最小化, FPS 最大化)。"""
    if not points:
        return []
    front: List[Tuple[float, float]] = []
    for i, (e1, f1) in enumerate(points):
        dominated = False
        for j, (e2, f2) in enumerate(points):
            if i == j:
                continue
            if e2 <= e1 and f2 >= f1 and (e2 < e1 or f2 > f1):
                dominated = True
                break
        if not dominated:
            front.append((e1, f1))
    return front


# ---------------------------------------------------------------------------
# 聚合函数：每代结束时一次性算齐所有指标
# ---------------------------------------------------------------------------

def compute_full_generation_metrics(
    *,
    history_df: pd.DataFrame,
    current_population_arch_codes: Sequence[str],
    metrics_history_df: Optional[pd.DataFrame],
    epoch: int,
    new_evaluated: int,
    duplicates: int,
    population_size: int,
    search_space_size: int,
    ref_epe: float = DEFAULT_HV_REF_EPE,
    ref_fps: float = DEFAULT_HV_REF_FPS,
) -> Dict[str, Any]:
    """计算单代全套监控指标，返回符合 GENERATION_METRICS_COLUMNS 的 dict。

    Args:
        history_df: 当前实验全量 history_archive.csv
        current_population_arch_codes: 当前种群的 arch_code 列表（NSGA-II 选出的
            下一代父代，通常 50 个）
        metrics_history_df: 之前的 generation_metrics.csv（用于 stagnation 和
            HV improvement rate）；首代可以传 None 或空 DataFrame
        epoch: 当前代号
        new_evaluated: 本代实际新评估的子网数
        duplicates: 本代生成时被去重过滤掉的子网数
        population_size: NSGA-II 种群大小
        search_space_size: 搜索空间总大小（覆盖率分母）
        ref_epe / ref_fps: HV 参考点

    Returns:
        dict ready for csv writer with 36 columns matching GENERATION_METRICS_COLUMNS.
    """
    # === 基础计数 ===
    total = len(history_df)
    pop_size = max(1, int(population_size))
    duplicate_rate = float(duplicates) / pop_size

    # === Pareto 前沿（基于全量历史）===
    pareto_points: List[Tuple[float, float]] = []
    if not history_df.empty and "epe" in history_df.columns and "fps" in history_df.columns:
        try:
            valid = history_df.dropna(subset=["epe", "fps"]).copy()
            valid["epe"] = pd.to_numeric(valid["epe"], errors="coerce")
            valid["fps"] = pd.to_numeric(valid["fps"], errors="coerce")
            valid = valid.dropna(subset=["epe", "fps"])
            all_points = list(zip(valid["epe"].tolist(), valid["fps"].tolist()))
            pareto_points = _compute_pareto_front_2d(all_points)
        except Exception:
            pareto_points = []

    pareto_count = len(pareto_points)

    # === best EPE / best FPS ===
    best_epe_val: Any = ""
    best_fps_val: Any = ""
    if pareto_points:
        best_epe_val = round(min(p[0] for p in pareto_points), 6)
        best_fps_val = round(max(p[1] for p in pareto_points), 6)

    # === HV ===
    hv_value = round(hypervolume_2d(pareto_points, ref_epe=ref_epe, ref_fps=ref_fps), 6)

    # === rank-1 saturation：基于当前种群（不是历史）===
    rank1_saturation = 0.0
    if current_population_arch_codes and not history_df.empty:
        # 用 history 里的 (epe, fps) 给当前种群每个 arch 找指标
        arch_to_metrics: Dict[str, Tuple[float, float]] = {}
        try:
            tmp = history_df.dropna(subset=["epe", "fps"]).copy()
            tmp["epe"] = pd.to_numeric(tmp["epe"], errors="coerce")
            tmp["fps"] = pd.to_numeric(tmp["fps"], errors="coerce")
            tmp = tmp.dropna(subset=["epe", "fps"])
            for _, row in tmp.iterrows():
                arch_to_metrics[str(row["arch_code"])] = (
                    float(row["epe"]), float(row["fps"]),
                )
        except Exception:
            pass

        pop_points: List[Tuple[float, float]] = []
        for ac in current_population_arch_codes:
            if str(ac) in arch_to_metrics:
                pop_points.append(arch_to_metrics[str(ac)])
        if pop_points:
            pop_first_front = _compute_pareto_front_2d(pop_points)
            rank1_saturation = round(len(pop_first_front) / max(1, len(pop_points)), 4)

    # === 平均拥挤距离 (前沿层面)===
    crowding = round(mean_crowding_distance_excluding_inf(pareto_points), 6)

    # === 每维基因熵 (当前种群层面)===
    entropies = per_dim_gene_entropy(current_population_arch_codes, num_dims=11)
    entropy_cols: Dict[str, float] = {
        f"gene_entropy_dim_{i}": round(entropies[i], 6) for i in range(11)
    }

    # === 最大 Pareto gap ===
    gap_info = largest_pareto_gap(pareto_points)

    # === Stagnation + HV improvement rate + duplicate_rate 滑动均值 ===
    stag_epe = 0
    stag_fps = 0
    stag_hv = 0
    hv_imp_rate: Any = ""
    duplicate_rate_3gen_avg = round(duplicate_rate, 6)

    history_present = (
        metrics_history_df is not None
        and not getattr(metrics_history_df, "empty", True)
    )
    if history_present:
        # best_epe stagnation
        if "best_epe" in metrics_history_df.columns:
            past = pd.to_numeric(
                metrics_history_df["best_epe"], errors="coerce"
            ).dropna().tolist()
            seq = past + ([float(best_epe_val)] if best_epe_val != "" else [])
            stag_epe = stagnation_count(seq, direction="decrease")
        # best_fps stagnation
        if "best_fps" in metrics_history_df.columns:
            past = pd.to_numeric(
                metrics_history_df["best_fps"], errors="coerce"
            ).dropna().tolist()
            seq = past + ([float(best_fps_val)] if best_fps_val != "" else [])
            stag_fps = stagnation_count(seq, direction="increase")
        # HV stagnation + improvement rate
        if "hv" in metrics_history_df.columns:
            past = pd.to_numeric(
                metrics_history_df["hv"], errors="coerce"
            ).dropna().tolist()
            seq = past + [hv_value]
            stag_hv = stagnation_count(seq, direction="increase")
            if len(seq) >= 4:
                prev_hv = seq[-4]
                hv_imp_rate = round((hv_value - prev_hv) / 3.0, 6)
        # duplicate_rate 滑动均值
        if "duplicate_rate" in metrics_history_df.columns:
            past = pd.to_numeric(
                metrics_history_df["duplicate_rate"], errors="coerce"
            ).dropna().tolist()
            seq = past + [duplicate_rate]
            window = seq[-3:]
            if window:
                duplicate_rate_3gen_avg = round(sum(window) / len(window), 6)

    coverage = round(total / max(1, int(search_space_size)) * 100, 2)

    metrics: Dict[str, Any] = {
        "epoch": int(epoch),
        "total_evaluated": int(total),
        "new_evaluated": int(new_evaluated),
        "duplicates": int(duplicates),
        "duplicate_rate": round(duplicate_rate, 4),
        "duplicate_rate_3gen_avg": duplicate_rate_3gen_avg,
        "rule_rejected": 0,
        "best_epe": best_epe_val,
        "best_fps": best_fps_val,
        "pareto_count": int(pareto_count),
        "rank1_saturation": rank1_saturation,
        "hv": hv_value,
        "hv_ref_epe": float(ref_epe),
        "hv_ref_fps": float(ref_fps),
        "hv_improvement_rate_3gen": hv_imp_rate,
        "mean_crowding_distance": crowding,
        **entropy_cols,
        "stagnation_best_epe": int(stag_epe),
        "stagnation_best_fps": int(stag_fps),
        "stagnation_hv": int(stag_hv),
        "largest_gap_fps_low": gap_info["fps_low"],
        "largest_gap_fps_high": gap_info["fps_high"],
        "largest_gap_epe_low": gap_info["epe_low"],
        "largest_gap_epe_high": gap_info["epe_high"],
        "findings_count": 0,
        "assumptions_count": 0,
        "coverage_pct": coverage,
    }
    return metrics
