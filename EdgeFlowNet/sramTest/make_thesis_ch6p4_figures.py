"""
Generate fig_6p3_hypervolume_geometry.pdf for Ch6.4.

Pedagogical figure: shows what hypervolume is geometrically.
- All 800 evaluated subnets as light grey dots in (EPE, FPS) space
- Pareto front as dark blue markers connected by a staircase
- Reference point (5.5, 3.0) as a red star
- Hypervolume = shaded area between the staircase and the reference point

Uses the final-state evaluation pool of group d_v3 (the full hybrid),
as a representative single-run snapshot. The numerical HV value matches
the chapter-end ablation.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
_HISTORY = (
    _REPO / "EdgeFlowNAS" / "outputs" / "ablation_phase5"
    / "group_d_20260506_192100" / "metadata" / "history_archive.csv"
)
_OUT_FIG = _REPO / "figures" / "fig_6p3_hypervolume_geometry.pdf"

_REF_EPE = 5.5
_REF_FPS = 3.0


def _load_points(path: Path) -> List[Tuple[float, float]]:
    pts = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                pts.append((float(row["epe"]), float(row["fps"])))
            except (KeyError, ValueError, TypeError):
                continue
    return pts


def _pareto_front(points, ref_epe, ref_fps):
    def is_dominated(p, others):
        e, fps = p
        for oe, of in others:
            if (oe, of) == p:
                continue
            if oe <= e and of >= fps and (oe < e or of > fps):
                return True
        return False
    in_box = [p for p in points if p[0] <= ref_epe and p[1] >= ref_fps]
    return [p for p in in_box if not is_dominated(p, in_box)]


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "axes.labelweight": "bold",
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.grid": True,
            "grid.linestyle": "-",
            "grid.linewidth": 0.5,
            "grid.color": "#e6e6e6",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.9,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def main() -> None:
    pts = _load_points(_HISTORY)
    # Sort by FPS ascending so the staircase walks left-to-right.
    pareto = sorted(_pareto_front(pts, _REF_EPE, _REF_FPS), key=lambda p: p[1])
    print(f"Total evals: {len(pts)}, Pareto-in-box: {len(pareto)}")

    _configure_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    # Layout: X = FPS (higher is better), Y = EPE (lower is better)
    # Reference point at (FPS=3.0, EPE=5.5) sits in the top-left of the ref box;
    # Pareto staircase walks from lower-left (low FPS, low EPE) up to upper-right.

    # 1. All evaluated points (light grey background): x=fps, y=epe
    xs = [p[1] for p in pts]
    ys = [p[0] for p in pts]
    ax.scatter(
        xs, ys,
        s=10, color="#bcbcbc", alpha=0.55,
        edgecolors="none", label=f"All {len(pts)} evaluations",
    )

    # 2. Hypervolume shaded area, decomposed into strips along FPS axis.
    # Strip 0 (leftmost): x in [ref_fps, fps_1], y in [epe_1, ref_epe]
    # Strip i for i>=1: x in [fps_i, fps_{i+1}], y in [epe_{i+1}, ref_epe]
    N = len(pareto)
    hv_total = 0.0
    if N > 0:
        epe_1, fps_1 = pareto[0]
        # Leftmost strip
        ax.fill_between(
            [_REF_FPS, fps_1], epe_1, _REF_EPE,
            color="#3b75af", alpha=0.16, edgecolor="none",
        )
        hv_total += (fps_1 - _REF_FPS) * (_REF_EPE - epe_1)
        # Remaining strips
        for i in range(N - 1):
            epe_i, fps_i = pareto[i]
            epe_next, fps_next = pareto[i + 1]
            ax.fill_between(
                [fps_i, fps_next], epe_next, _REF_EPE,
                color="#3b75af", alpha=0.16, edgecolor="none",
            )
            hv_total += (fps_next - fps_i) * (_REF_EPE - epe_next)
    print(f"Computed HV: {hv_total:.4f}")

    # 3. Pareto staircase line: start at (fps_1, epe_1), step up + right alternately
    if N > 0:
        step_x = [pareto[0][1]]  # fps_1
        step_y = [pareto[0][0]]  # epe_1
        for i in range(1, N):
            epe_prev, fps_prev = pareto[i - 1]
            epe_i, fps_i = pareto[i]
            # vertical jump up at fps_prev from epe_prev to epe_i
            step_x.append(fps_prev)
            step_y.append(epe_i)
            # horizontal walk right at epe_i to fps_i
            step_x.append(fps_i)
            step_y.append(epe_i)
        ax.plot(step_x, step_y, color="#3b75af", linewidth=1.3, zorder=2)

    # 4. Pareto front markers
    px = [p[1] for p in pareto]
    py = [p[0] for p in pareto]
    ax.scatter(
        px, py,
        s=44, color="#3b75af", edgecolors="white", linewidths=0.6,
        zorder=3, label=f"Pareto front ({N} points)",
    )

    # 5. Reference point
    ax.scatter(
        [_REF_FPS], [_REF_EPE],
        marker="*", s=240, color="#d62728",
        edgecolors="white", linewidths=0.8, zorder=4,
        label=f"Reference point ({_REF_FPS}, {_REF_EPE})",
    )

    # 6. Annotation inside the shaded HV region
    ax.annotate(
        f"Hypervolume\n= shaded area\n$\\approx {hv_total:.3f}$",
        xy=(7.5, 5.1), xytext=(7.5, 5.1),
        ha="center", va="center",
        fontsize=12, color="#1f4e7c",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#3b75af", alpha=0.92, linewidth=0.8),
    )

    # 7. Labels and axis range
    ax.set_xlabel("FPS  (higher is better)")
    ax.set_ylabel("EPE  (lower is better)")
    ax.set_xlim(2.75, 9.7)
    ax.set_ylim(3.95, 5.6)
    ax.legend(loc="lower right", framealpha=0.92)

    _OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT_FIG, format="pdf")
    print(f"Wrote {_OUT_FIG}")


if __name__ == "__main__":
    main()
