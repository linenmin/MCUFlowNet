"""
Generate fig_6p3_hv_trajectory.pdf for Ch6.5 Ablation.

Three configurations compared:
  - group_a: NSGA-II baseline (no LLM agents)
  - group_b: + Warmstart agent
  - group_d_v3: full hybrid system (Warmstart + Scientist + Supervisor)

The figure shows hypervolume vs evaluation budget for the first 500 evals
(10 generations of 50 subnets each).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
_OUTPUTS = _REPO / "EdgeFlowNAS" / "outputs"
_NSGA_V3_DISTILL = (
    _OUTPUTS / "nsga2_v3" / "nsga2_v3_distill_run1_20260429_201744"
    / "metadata" / "history_archive.csv"
)
_GROUP_B = (
    _OUTPUTS / "search_hybrid_v2" / "group_b_v4_thinking_20260521_171443"
    / "metadata" / "epoch_metrics.csv"
)
_GROUP_D_V3 = (
    _OUTPUTS / "search_hybrid_v2" / "group_d_v4_thinking_20260522_161253"
    / "metadata" / "epoch_metrics.csv"
)
_OUT_FIG = _REPO / "figures" / "fig_6p3_hv_trajectory.pdf"

_REF_EPE = 5.5
_REF_FPS = 3.0
_POP_PER_GEN = 50
_MAX_EPOCH = 15  # epochs 0..15 inclusive = 16 generations = 800 evaluations


def _compute_hv_per_epoch(path: Path) -> List[float]:
    """Compute cumulative HV per epoch from history_archive (no HV column)."""
    pts: List[Tuple[int, float, float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                pts.append((int(row["epoch"]), float(row["epe"]), float(row["fps"])))
            except (KeyError, ValueError, TypeError):
                continue

    def is_dominated(p, others):
        e, fps = p
        for oe, of in others:
            if (oe, of) == p:
                continue
            if oe <= e and of >= fps and (oe < e or of > fps):
                return True
        return False

    hvs: List[float] = []
    for ep in range(_MAX_EPOCH + 1):
        cumulative = [(e, f) for (ee, e, f) in pts if ee <= ep]
        pareto = [p for p in cumulative if not is_dominated(p, cumulative)]
        in_box = sorted(
            [p for p in pareto if p[0] <= _REF_EPE and p[1] >= _REF_FPS],
            key=lambda p: p[1],
            reverse=True,
        )
        hv = 0.0
        prev_epe = _REF_EPE
        for epe, fps in in_box:
            if epe < prev_epe:
                hv += (prev_epe - epe) * (fps - _REF_FPS)
                prev_epe = epe
        hvs.append(hv)
    return hvs


def _read_hv_column(path: Path) -> List[float]:
    """Read the per-epoch HV column from epoch_metrics.csv."""
    hvs: List[float] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                ep = int(row["epoch"])
            except (KeyError, ValueError):
                continue
            if ep > _MAX_EPOCH:
                break
            hvs.append(float(row["hv"]))
    return hvs


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "axes.labelweight": "bold",
            "legend.fontsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.grid": True,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "grid.color": "#e2e6ea",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.9,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def main() -> None:
    hv_a = _compute_hv_per_epoch(_NSGA_V3_DISTILL)
    hv_b = _read_hv_column(_GROUP_B)
    hv_d = _read_hv_column(_GROUP_D_V3)
    n = min(len(hv_a), len(hv_b), len(hv_d))
    xs = [(ep + 1) * _POP_PER_GEN for ep in range(n)]

    _configure_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    # Pareto-coverage ceiling: max final HV observed across runs
    ceiling = max(hv_a[-1], hv_b[-1], hv_d[-1])
    ax.axhline(
        y=ceiling,
        color="#666666",
        linestyle="--",
        linewidth=1.2,
        alpha=0.65,
    )
    ax.text(
        x=_POP_PER_GEN * (_MAX_EPOCH + 1) - 5,
        y=ceiling + 0.04,
        s=f"Pareto-coverage ceiling ($\\approx {ceiling:.3f}$)",
        ha="right",
        va="bottom",
        fontsize=11,
        color="#555555",
        style="italic",
    )

    ax.plot(
        xs, hv_a[:n],
        label="C1: NSGA-II baseline",
        color="#3b75af", marker="o", markersize=5, linewidth=1.9,
    )
    ax.plot(
        xs, hv_b[:n],
        label="C2: + Warm-start",
        color="#ef8636", marker="s", markersize=5, linewidth=1.9,
    )
    ax.plot(
        xs, hv_d[:n],
        label="C3: + Warm-start + Scientist + Supervisor",
        color="#3a923a", marker="^", markersize=5.5, linewidth=1.9,
    )

    # Recovery-story annotation: (c) starts 0.243 HV below (b) at evaluation 50
    # and recovers to within 0.0014 HV of (b) by evaluation 800.
    gen0_evals = _POP_PER_GEN
    gen0_b = hv_b[0]
    gen0_d = hv_d[0]
    # Vertical bracket between (b) and (c) gen-0 points.
    ax.annotate(
        "",
        xy=(gen0_evals + 6, gen0_d),
        xytext=(gen0_evals + 6, gen0_b),
        arrowprops=dict(
            arrowstyle="-",
            color="#7a7a7a",
            linewidth=1.0,
        ),
    )
    ax.text(
        gen0_evals + 16,
        (gen0_b + gen0_d) / 2,
        "C3 starts 0.243\nHV below C2",
        ha="left",
        va="center",
        fontsize=10,
        color="#3a923a",
        style="italic",
    )
    ax.text(
        _POP_PER_GEN * (_MAX_EPOCH + 1) - 60,
        7.485,
        "C3 recovers to within\n0.0014 HV of C2 by 800 evals",
        ha="right",
        va="top",
        fontsize=10,
        color="#3a923a",
        style="italic",
    )

    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Hypervolume")
    ax.set_xlim(left=0, right=_POP_PER_GEN * (_MAX_EPOCH + 1) + 5)
    ax.set_xticks([0, 100, 200, 300, 400, 500, 600, 700, 800])
    ax.legend(loc="lower right", framealpha=0.9)

    _OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT_FIG, format="pdf")
    print(f"Wrote {_OUT_FIG}")


if __name__ == "__main__":
    main()
