"""Phase 5 (search_hybrid_v1): Ablation 后处理分析脚本.

读 4 组 ablation 跑的 history_archive.csv, 计算关键指标 + 输出对比图表 / JSON
摘要 / markdown 报告. 红线判定也在这里出.

用法:
    python wrappers/run_phase5_ablation_analysis.py \\
        --output_dir outputs/ablation_phase5/analysis_$(date +%Y%m%d_%H%M%S)

默认从下面的 ABLATION_RUNS dict 读 4 组路径; CLI 可 --override_path 单独指定.
"""

import argparse
import csv
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from efnas.search.search_metrics import (
    DEFAULT_HV_REF_EPE,
    DEFAULT_HV_REF_FPS,
    _compute_pareto_front_2d,
    hypervolume_2d,
)

logger = logging.getLogger("phase5_ablation_analysis")


# ============================================================
# 默认 ablation 数据源 (CLI 可覆盖)
# ============================================================
ABLATION_RUNS: Dict[str, str] = {
    "a": "outputs/nsga2_v3/nsga2_v3_distill_run1_20260429_201744/metadata/history_archive.csv",
    "b": "outputs/ablation_phase5/group_b/metadata/history_archive.csv",
    "c": "outputs/ablation_phase5/group_c/metadata/history_archive.csv",
    "d": "outputs/ablation_phase5/group_d/metadata/history_archive.csv",
}

GROUP_LABELS = {
    "a": "(a) NSGA-II only",
    "b": "(b) + Warmstart",
    "c": "(c) + Scientist",
    "d": "(d) + Supervisor (full)",
}

# Phase 5 比较的预算点
EVAL_BUDGETS: Sequence[int] = (50, 100, 200, 400, 600, 800)


# ============================================================
# 单组 ablation cell 的指标计算
# ============================================================

def load_history(path: str) -> pd.DataFrame:
    """读 history_archive.csv, 数值列做强制类型转换."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"history_archive.csv not found: {path}")
    df = pd.read_csv(path, dtype={"arch_code": str})
    # NSGA-II 的"评估顺序"由 timestamp 升序定义 (epoch 列只是粗粒度)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    df["epe"] = pd.to_numeric(df["epe"], errors="coerce")
    df["fps"] = pd.to_numeric(df["fps"], errors="coerce")
    df = df.dropna(subset=["epe", "fps"]).reset_index(drop=True)
    return df


def compute_trajectory(
    history_df: pd.DataFrame,
    budgets: Sequence[int],
    *,
    ref_epe: float = DEFAULT_HV_REF_EPE,
    ref_fps: float = DEFAULT_HV_REF_FPS,
) -> Dict[str, List[float]]:
    """对历史按评估顺序切片到不同预算点, 计算每个点的 HV / best_epe / best_fps /
    pareto_count.

    Returns:
        dict 各字段是 List[float], 顺序对应 budgets.
    """
    n = len(history_df)
    hv_traj: List[float] = []
    best_epe_traj: List[float] = []
    best_fps_traj: List[float] = []
    pareto_count_traj: List[int] = []
    for budget in budgets:
        eff_budget = min(budget, n)
        sub = history_df.iloc[:eff_budget]
        if len(sub) == 0:
            hv_traj.append(0.0)
            best_epe_traj.append(float("nan"))
            best_fps_traj.append(float("nan"))
            pareto_count_traj.append(0)
            continue
        points = list(zip(sub["epe"].tolist(), sub["fps"].tolist()))
        front = _compute_pareto_front_2d(points)
        hv = hypervolume_2d(front, ref_epe=ref_epe, ref_fps=ref_fps)
        hv_traj.append(round(hv, 6))
        best_epe_traj.append(round(float(sub["epe"].min()), 6))
        best_fps_traj.append(round(float(sub["fps"].max()), 6))
        pareto_count_traj.append(len(front))
    return {
        "budgets": list(budgets),
        "hv": hv_traj,
        "best_epe": best_epe_traj,
        "best_fps": best_fps_traj,
        "pareto_count": pareto_count_traj,
    }


def compute_pareto_summary(history_df: pd.DataFrame) -> Dict[str, Any]:
    """对完整历史的 Pareto 前沿做端点 + 范围统计."""
    if history_df.empty:
        return {}
    points = list(zip(history_df["epe"].tolist(), history_df["fps"].tolist()))
    front = _compute_pareto_front_2d(points)
    if not front:
        return {"pareto_count": 0}
    epes = [p[0] for p in front]
    fpss = [p[1] for p in front]
    best_epe_idx = epes.index(min(epes))
    best_fps_idx = fpss.index(max(fpss))
    return {
        "pareto_count": len(front),
        "epe_range": [round(min(epes), 6), round(max(epes), 6)],
        "fps_range": [round(min(fpss), 6), round(max(fpss), 6)],
        "best_epe_point": {
            "epe": round(epes[best_epe_idx], 6),
            "fps": round(fpss[best_epe_idx], 6),
        },
        "best_fps_point": {
            "epe": round(epes[best_fps_idx], 6),
            "fps": round(fpss[best_fps_idx], 6),
        },
    }


# ============================================================
# 红线判定
# ============================================================

def evaluate_red_lines(group_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """对照 task_plan.md 红线: 每加一个 phase 的 final HV 必须 >= 上一阶段."""
    final_hv: Dict[str, float] = {}
    for g, res in group_results.items():
        traj = res.get("trajectory", {})
        hv_seq = traj.get("hv", [])
        final_hv[g] = float(hv_seq[-1]) if hv_seq else 0.0

    verdicts: Dict[str, str] = {}
    if "a" in final_hv and "b" in final_hv:
        if final_hv["b"] >= final_hv["a"]:
            verdicts["phase2_warmstart"] = (
                f"PASS: HV({final_hv['b']:.4f}) >= baseline HV({final_hv['a']:.4f})"
            )
        else:
            verdicts["phase2_warmstart"] = (
                f"FAIL: HV({final_hv['b']:.4f}) < baseline HV({final_hv['a']:.4f}) "
                f"-- consider removing Phase 2"
            )
    if "b" in final_hv and "c" in final_hv:
        if final_hv["c"] >= final_hv["b"]:
            verdicts["phase3_scientist"] = (
                f"PASS: HV({final_hv['c']:.4f}) >= warmstart HV({final_hv['b']:.4f})"
            )
        else:
            verdicts["phase3_scientist"] = (
                f"FAIL: HV({final_hv['c']:.4f}) < warmstart HV({final_hv['b']:.4f}) "
                f"-- consider removing Phase 3"
            )
    if "c" in final_hv and "d" in final_hv:
        if final_hv["d"] >= final_hv["c"]:
            verdicts["phase4_supervisor"] = (
                f"PASS: HV({final_hv['d']:.4f}) >= scientist HV({final_hv['c']:.4f})"
            )
        else:
            verdicts["phase4_supervisor"] = (
                f"FAIL: HV({final_hv['d']:.4f}) < scientist HV({final_hv['c']:.4f}) "
                f"-- Phase 4 should be ablated out"
            )
    return {
        "final_hv": final_hv,
        "verdicts": verdicts,
    }


# ============================================================
# Plots
# ============================================================

def plot_hv_trajectories(
    group_results: Dict[str, Dict[str, Any]],
    output_path: str,
) -> None:
    """画各组 HV vs evaluation 数的折线对比图."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    color_map = {"a": "#4C72B0", "b": "#DD8452", "c": "#55A868", "d": "#C44E52"}
    for group in ("a", "b", "c", "d"):
        if group not in group_results:
            continue
        traj = group_results[group].get("trajectory", {})
        if not traj:
            continue
        ax.plot(
            traj["budgets"], traj["hv"],
            marker="o", linewidth=2,
            color=color_map.get(group),
            label=GROUP_LABELS.get(group, group),
        )
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Hypervolume (HV)")
    ax.set_title("Phase 5 Ablation: HV trajectory across ablation groups")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("HV trajectory plot saved: %s", output_path)


def plot_pareto_fronts(
    group_results: Dict[str, Dict[str, Any]],
    output_path: str,
) -> None:
    """画各组最终 Pareto 前沿在 (FPS, EPE) 空间的散点对比."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    color_map = {"a": "#4C72B0", "b": "#DD8452", "c": "#55A868", "d": "#C44E52"}
    for group in ("a", "b", "c", "d"):
        if group not in group_results:
            continue
        df = group_results[group].get("history_df")
        if df is None or df.empty:
            continue
        points = list(zip(df["epe"].tolist(), df["fps"].tolist()))
        front = _compute_pareto_front_2d(points)
        if not front:
            continue
        front_sorted = sorted(front, key=lambda p: p[1])
        front_fps = [p[1] for p in front_sorted]
        front_epe = [p[0] for p in front_sorted]
        ax.plot(
            front_fps, front_epe,
            marker="o", linewidth=1.5,
            color=color_map.get(group),
            label=GROUP_LABELS.get(group, group),
        )
    ax.set_xlabel("FPS (higher is better)")
    ax.set_ylabel("EPE (lower is better)")
    ax.set_title("Phase 5 Ablation: Final Pareto fronts (V3, distill supernet)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Pareto fronts plot saved: %s", output_path)


# ============================================================
# Markdown 报告
# ============================================================

def write_markdown_report(
    group_results: Dict[str, Dict[str, Any]],
    red_lines: Dict[str, Any],
    output_path: str,
) -> None:
    lines: List[str] = []
    lines.append("# Phase 5 Ablation Analysis Report")
    lines.append("")
    lines.append(f"Generated: {pd.Timestamp.now().isoformat()}")
    lines.append("")

    # 各组数据源
    lines.append("## Data sources")
    lines.append("")
    lines.append("| Group | Path | n_evals |")
    lines.append("|---|---|---|")
    for g in ("a", "b", "c", "d"):
        if g not in group_results:
            continue
        n = group_results[g].get("n_evals", 0)
        path = group_results[g].get("path", "")
        lines.append(f"| {GROUP_LABELS.get(g, g)} | `{path}` | {n} |")
    lines.append("")

    # HV 表
    lines.append("## HV at evaluation budgets")
    lines.append("")
    budget_header = " | ".join(str(b) for b in EVAL_BUDGETS)
    lines.append(f"| Group | {budget_header} |")
    lines.append("|" + "---|" * (len(EVAL_BUDGETS) + 1))
    for g in ("a", "b", "c", "d"):
        if g not in group_results:
            continue
        traj = group_results[g].get("trajectory", {})
        hv_seq = traj.get("hv", [])
        cells = " | ".join(f"{x:.4f}" for x in hv_seq)
        lines.append(f"| {GROUP_LABELS.get(g, g)} | {cells} |")
    lines.append("")

    # best_epe / best_fps 端点表
    lines.append("## Final Pareto front summary")
    lines.append("")
    lines.append("| Group | Pareto count | EPE range | FPS range | "
                 "Best EPE point (EPE/FPS) | Best FPS point (EPE/FPS) |")
    lines.append("|---|---|---|---|---|---|")
    for g in ("a", "b", "c", "d"):
        if g not in group_results:
            continue
        s = group_results[g].get("pareto_summary", {})
        if not s:
            continue
        best_epe = s.get("best_epe_point", {})
        best_fps = s.get("best_fps_point", {})
        lines.append(
            f"| {GROUP_LABELS.get(g, g)} | "
            f"{s.get('pareto_count', 0)} | "
            f"{s.get('epe_range', ['?', '?'])} | "
            f"{s.get('fps_range', ['?', '?'])} | "
            f"{best_epe.get('epe', '?'):.4f} / {best_epe.get('fps', '?'):.4f} | "
            f"{best_fps.get('epe', '?'):.4f} / {best_fps.get('fps', '?'):.4f} |"
        )
    lines.append("")

    # 红线判定
    lines.append("## Red line verdicts (vs task_plan.md success criteria)")
    lines.append("")
    final_hv = red_lines.get("final_hv", {})
    if final_hv:
        lines.append("Final HV per group:")
        lines.append("")
        for g, hv in final_hv.items():
            lines.append(f"- {GROUP_LABELS.get(g, g)}: **{hv:.4f}**")
        lines.append("")
    verdicts = red_lines.get("verdicts", {})
    for phase_name, verdict in verdicts.items():
        marker = "✅" if verdict.startswith("PASS") else "❌"
        lines.append(f"- **{phase_name}**: {marker} {verdict}")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- HV reference point: "
        f"`(EPE_ref={DEFAULT_HV_REF_EPE}, FPS_ref={DEFAULT_HV_REF_FPS})`"
    )
    lines.append("- Single seed per cell (no variance estimate). FAIL verdicts")
    lines.append("  with small margins should be re-run with additional seeds")
    lines.append("  before declaring a Phase removal.")
    lines.append("- Group (a) is reused from `outputs/nsga2_v3/nsga2_v3_distill_run1_*`")
    lines.append("  pre-Phase-1 NSGA-II baseline (compatible: only history_archive.csv")
    lines.append("  is consumed; richer Phase 1.3 metrics not required for ablation).")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Markdown report saved: %s", output_path)


# ============================================================
# Main
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 5 ablation analysis (search_hybrid_v1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="output directory for plots/json/md; default: "
             "outputs/ablation_phase5/analysis_<timestamp>",
    )
    for group in ("a", "b", "c", "d"):
        parser.add_argument(
            f"--path_{group}",
            type=str,
            default=None,
            help=f"override history_archive.csv path for group ({group})",
        )
    parser.add_argument(
        "--ref_epe", type=float, default=DEFAULT_HV_REF_EPE,
        help=f"HV EPE reference point (default {DEFAULT_HV_REF_EPE})",
    )
    parser.add_argument(
        "--ref_fps", type=float, default=DEFAULT_HV_REF_FPS,
        help=f"HV FPS reference point (default {DEFAULT_HV_REF_FPS})",
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _build_parser().parse_args()

    if args.output_dir is None:
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(
            "outputs", "ablation_phase5", f"analysis_{ts}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # 解析每组路径
    paths: Dict[str, str] = {}
    for group, default_path in ABLATION_RUNS.items():
        cli_override = getattr(args, f"path_{group}", None)
        path = cli_override if cli_override else default_path
        if not os.path.isabs(path):
            path = os.path.join(_PROJECT_ROOT, path)
        paths[group] = path

    # 加载 + 计算
    group_results: Dict[str, Dict[str, Any]] = {}
    for group, path in paths.items():
        if not os.path.exists(path):
            logger.warning(
                "[%s] history_archive.csv 不存在, 跳过此组: %s", group, path,
            )
            continue
        try:
            df = load_history(path)
        except Exception:
            logger.exception("[%s] load_history 失败", group)
            continue
        traj = compute_trajectory(
            df, EVAL_BUDGETS, ref_epe=args.ref_epe, ref_fps=args.ref_fps,
        )
        pareto_summary = compute_pareto_summary(df)
        group_results[group] = {
            "path": path,
            "n_evals": len(df),
            "trajectory": traj,
            "pareto_summary": pareto_summary,
            "history_df": df,  # in-memory only, 不进 JSON
        }
        logger.info(
            "[%s] loaded %d evaluations, final HV = %.4f",
            group, len(df), traj["hv"][-1] if traj["hv"] else 0.0,
        )

    if not group_results:
        logger.error("没有任何组成功加载 history_archive.csv. 跳过分析.")
        return 1

    red_lines = evaluate_red_lines(group_results)

    # 写 JSON 摘要 (排除 in-memory DataFrame)
    summary_payload = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "ref_point": {"epe": args.ref_epe, "fps": args.ref_fps},
        "groups": {
            g: {
                "path": r["path"],
                "n_evals": r["n_evals"],
                "trajectory": r["trajectory"],
                "pareto_summary": r["pareto_summary"],
            }
            for g, r in group_results.items()
        },
        "red_lines": red_lines,
    }
    json_path = os.path.join(args.output_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)
    logger.info("JSON summary saved: %s", json_path)

    # 画图
    try:
        plot_hv_trajectories(
            group_results, os.path.join(args.output_dir, "hv_trajectory.png"),
        )
    except Exception:
        logger.exception("HV trajectory 画图失败")
    try:
        plot_pareto_fronts(
            group_results, os.path.join(args.output_dir, "pareto_fronts.png"),
        )
    except Exception:
        logger.exception("Pareto fronts 画图失败")

    # Markdown 报告
    md_path = os.path.join(args.output_dir, "report.md")
    write_markdown_report(group_results, red_lines, md_path)

    logger.info("=" * 60)
    logger.info("Phase 5 ablation analysis finished")
    logger.info("Output dir: %s", args.output_dir)
    logger.info("=" * 60)

    # Print final HV summary to stdout
    print()
    print("=== FINAL HV PER GROUP ===")
    for g, hv in red_lines.get("final_hv", {}).items():
        print(f"  {GROUP_LABELS.get(g, g)}: {hv:.4f}")
    print()
    print("=== RED LINE VERDICTS ===")
    for phase_name, verdict in red_lines.get("verdicts", {}).items():
        print(f"  {phase_name}: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
