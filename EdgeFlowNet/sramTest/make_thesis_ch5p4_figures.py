"""
Generate fig_5p4_subnet_rank_trajectory.pdf for Ch5.4 Supernet training.

Bump chart: the 12 stratified eval-pool subnets and how their inherited-weight
EPE rank changes across the 200-epoch supernet training. Each subnet is one
coloured line. The y axis is inverted: rank 1 (best EPE among the 12) sits at
the top. Visual claim: many rank crossings early in training, fewer crossings
later, and the final three evaluations (epochs 190, 195, 200) return the same
12-subnet order (parallel lines in the right-hand shaded region).

Data source (relative to repo root):
  EdgeFlowNAS/outputs/supernet/
  edgeflownas_supernet_v3_fc2_172x224_run1_archparallel_distill/
  eval_epe_history.csv
Column `arch_rank_12` format per row: "rank:subnet_id:epe|rank:subnet_id:epe|..."
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.cm as cm

_REPO = Path(__file__).resolve().parents[2]
_HISTORY = (
    _REPO / "EdgeFlowNAS" / "outputs" / "supernet"
    / "edgeflownas_supernet_v3_fc2_172x224_run1_archparallel_distill"
    / "eval_epe_history.csv"
)
_OUT_FIG = _REPO / "figures" / "fig_5p4_subnet_rank_trajectory.pdf"

_N_SUBNETS = 12
# Low-change shading: per-evaluation rank-change count drops to 2-4 movements
# from epoch 175 onward; the final two evaluations at epochs 195 and 200
# return identical 12-subnet orderings (strict-invariance window of 10 epochs).
_CONVERGED_FROM_EPOCH = 175


def _parse_rank_column(cell: str) -> Dict[int, int]:
    """Parse 'rank:subnet_id:epe|...' into {subnet_id: rank}."""
    out: Dict[int, int] = {}
    for piece in cell.split("|"):
        parts = piece.split(":")
        if len(parts) < 3:
            continue
        rank = int(parts[0])
        subnet_id = int(parts[1])
        out[subnet_id] = rank
    return out


def _load_trajectories(path: Path) -> tuple[List[int], Dict[int, List[int]]]:
    """Return (epochs, {subnet_id: [rank_at_epoch_i]})."""
    epochs: List[int] = []
    by_subnet: Dict[int, List[int]] = {sid: [] for sid in range(_N_SUBNETS)}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                ep = int(row["epoch"])
            except (KeyError, ValueError):
                continue
            ranks = _parse_rank_column(row["arch_rank_12"])
            if len(ranks) != _N_SUBNETS:
                continue
            epochs.append(ep)
            for sid in range(_N_SUBNETS):
                by_subnet[sid].append(ranks[sid])
    return epochs, by_subnet


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "axes.labelweight": "bold",
            "legend.fontsize": 11,
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
    epochs, by_subnet = _load_trajectories(_HISTORY)
    if not epochs:
        raise SystemExit(f"no usable rows in {_HISTORY}")

    _configure_style()
    fig, ax = plt.subplots(figsize=(9.0, 5.4))

    # Shaded "converged" region (epochs 135-200).
    converged_left = max(_CONVERGED_FROM_EPOCH, epochs[0])
    converged_right = epochs[-1]
    ax.axvspan(
        converged_left,
        converged_right,
        color="#e5edf7",
        alpha=0.55,
        zorder=0,
    )
    ax.text(
        (converged_left + converged_right) / 2,
        0.6,
        "low-change regime",
        ha="center",
        va="center",
        fontsize=11,
        color="#3b5b8a",
        style="italic",
        zorder=1,
    )

    # 12 perceptually distinct colours; only one green to avoid the two-greens
    # clash the tab20 default produced (green vs olive).
    colours = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#e7c000", "#17becf", "#393b79", "#f032e6",
    ]

    for sid in range(_N_SUBNETS):
        ranks = by_subnet[sid]
        ax.plot(
            epochs,
            ranks,
            color=colours[sid],
            marker="o",
            markersize=4.0,
            linewidth=1.6,
            alpha=0.88,
            zorder=2,
        )
        # Right-side end label with subnet ID.
        ax.text(
            epochs[-1] + 3,
            ranks[-1],
            f"S{sid}",
            ha="left",
            va="center",
            fontsize=10,
            color=colours[sid],
            fontweight="bold",
        )

    ax.set_xlabel("Supernet training epoch")
    ax.set_ylabel("Rank within 12-subnet eval pool")
    ax.set_xlim(left=0, right=epochs[-1] + 18)
    ax.set_xticks([0, 25, 50, 75, 100, 125, 150, 175, 200])
    ax.set_ylim(_N_SUBNETS + 0.5, 0.5)  # Inverted: rank 1 at top.
    ax.set_yticks(list(range(1, _N_SUBNETS + 1)))

    _OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT_FIG, format="pdf")
    print(f"Wrote {_OUT_FIG}")


if __name__ == "__main__":
    main()
