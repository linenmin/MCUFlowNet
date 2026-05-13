"""
Generate fig_4p6_ablation_curves.pdf for Ch4.4 Cumulative Ablation.

Thesis-local copy of plot_ablation_v1_fc2_epe_curves.py with:
  - variant labels stripped of the A0/A1/A2/A4/A3 experiment-order prefixes,
    so the figure reads as a clean ablation table (Deconv / Bilinear /
    Bilinear + ECA / Bilinear + Gate / Bilinear + ECA + Gate)
  - legend order follows the ablation logic, not the experiment order
  - larger fonts so the labels remain readable at thesis figure size

Inputs (already on disk):
  - D:/Dataset/MCUFlowNet/EdgeFlowNAS/outputs/ablation_v1_fc2/
    eval_history.csv files for the five trained variants.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt


_FC2_EPE_YLIM = (2.9, 3.8)

_FC2_METRIC = "best_epe"

_OUTPUTS = Path(r"D:/Dataset/MCUFlowNet/EdgeFlowNAS/outputs/ablation_v1_fc2")
_OUT_FIG = Path(
    r"D:/BaiduNetdiskWorkspace/Leuven/AI_Master_Thesis/thesis writing/Figure/"
    r"fig_4p6_ablation_curves.pdf"
)

_MAX_EPOCH = 130


@dataclass(frozen=True)
class VariantSpec:
    label: str
    csv_path: Path
    color: str
    linestyle: str
    marker: str


VARIANTS: List[VariantSpec] = [
    VariantSpec(
        label="Deconv",
        csv_path=_OUTPUTS
        / "ablation_v1_fc2_run1_p100_edgeflownet_deconv"
        / "model_edgeflownet_deconv"
        / "eval_history.csv",
        color="#3b75af",
        linestyle="-",
        marker="o",
    ),
    VariantSpec(
        label="Bilinear",
        csv_path=_OUTPUTS
        / "ablation_v1_fc2_run1_p100_edgeflownet_bilinear"
        / "model_edgeflownet_bilinear"
        / "eval_history.csv",
        color="#ef8636",
        linestyle="-",
        marker="o",
    ),
    VariantSpec(
        label="Bilinear + ECA",
        csv_path=_OUTPUTS
        / "ablation_v1_fc2"
        / "ablation_v1_fc2_run2_p100_skeleton_aligned"
        / "model_edgeflownet_bilinear_eca"
        / "eval_history.csv",
        color="#3a923a",
        linestyle="-",
        marker="o",
    ),
    VariantSpec(
        label="Bilinear + Gate",
        csv_path=_OUTPUTS
        / "ablation_v1_fc2"
        / "ablation_v1_fc2_run2_p100_skeleton_aligned_a4"
        / "model_edgeflownet_bilinear_gate4x"
        / "eval_history.csv",
        color="#8d69b8",
        linestyle="-",
        marker="o",
    ),
    VariantSpec(
        label="Bilinear + ECA + Gate",
        csv_path=_OUTPUTS
        / "ablation_v1_fc2"
        / "ablation_v1_fc2_run2_p100_skeleton_aligned"
        / "model_edgeflownet_bilinear_eca_gate4x"
        / "eval_history.csv",
        color="#e377c2",
        linestyle="-",
        marker="o",
    ),
]


def _read_history(csv_path: Path) -> Dict[str, List[float]]:
    epochs: List[float] = []
    fc2_best: List[float] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                epoch = float(row["epoch"])
            except (KeyError, ValueError, TypeError):
                continue
            if epoch > _MAX_EPOCH:
                continue
            epochs.append(epoch)
            fc2_best.append(float(row.get("best_epe") or "nan"))
    return {
        "epoch": epochs,
        "best_epe": fc2_best,
    }


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


def _draw(ax, xs, ys, ylim, spec: VariantSpec) -> None:
    if not xs:
        return
    low, high = ylim
    in_x = [x for x, y in zip(xs, ys) if y <= high and y == y]
    in_y = [y for y in ys if y <= high and y == y]
    over_x = [x for x, y in zip(xs, ys) if y == y and y > high]
    ax.plot(
        in_x, in_y,
        label=spec.label,
        color=spec.color, linestyle=spec.linestyle, marker=spec.marker,
        markersize=5.5, linewidth=1.9, alpha=0.95,
    )
    # off-scale eval points: render in neutral grey so they are not mistaken
    # for the colour-coded curves (they share the variant's epoch but the
    # actual EPE is above the panel cap).
    if over_x:
        ax.plot(
            over_x, [high] * len(over_x),
            linestyle="none", marker="^",
            markersize=8, color="#888888", alpha=0.55, clip_on=False,
        )


def _plot(records: Dict[str, Dict[str, List[float]]]) -> None:
    _configure_style()
    fig, ax_fc2 = plt.subplots(1, 1, figsize=(10.5, 5.2))

    for spec in VARIANTS:
        h = records[spec.label]
        _draw(ax_fc2, h["epoch"], h[_FC2_METRIC], _FC2_EPE_YLIM, spec)

    ax_fc2.set_title("Best FC2 validation EPE")
    ax_fc2.set_xlabel("Epoch")
    ax_fc2.set_ylabel("Best EPE")
    ax_fc2.set_ylim(*_FC2_EPE_YLIM)
    ax_fc2.set_xlim(0, _MAX_EPOCH)
    ax_fc2.tick_params(axis="both", which="both", length=0)
    # annotate the off-scale markers
    ax_fc2.text(
        _MAX_EPOCH * 0.98, _FC2_EPE_YLIM[1] - 0.05,
        r"$\blacktriangle$ off-scale eval (EPE above panel cap)",
        ha="right", va="top", fontsize=11, color="#666666",
    )

    handles, labels = ax_fc2.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(labels),
        frameon=False, bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))

    fig.savefig(_OUT_FIG)
    plt.close(fig)
    print(f"wrote {_OUT_FIG}")


def main() -> int:
    matplotlib.use("Agg")
    records = {spec.label: _read_history(spec.csv_path) for spec in VARIANTS}
    _plot(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
