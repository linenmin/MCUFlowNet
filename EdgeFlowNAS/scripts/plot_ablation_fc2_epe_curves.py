"""Plot Ablation V1 FC2 EPE curves (skeleton-aligned rerun + A4 + run1 references).

Produces SVG/PDF/PNG figures with FC2 val EPE and Sintel EPE vs epoch for
all five ablation variants:

  A0  edgeflownet_deconv                       (run1, transposed-conv)
  A1  edgeflownet_bilinear                     (run1, bilinear baseline)
  A2  edgeflownet_bilinear_eca                 (run2, skeleton-aligned)
  A4  edgeflownet_bilinear_gate4x              (run2_a4, gate only)
  A3  edgeflownet_bilinear_eca_gate4x          (run2, full proposal)

Output files (next to outputs/ablation_v1_fc2/):
  ablation_v1_fc2_epe_curves.svg
  ablation_v1_fc2_epe_curves_paper.pdf
  ablation_v1_fc2_epe_curves_paper.png
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt


# Y-axis clip windows. Tightened relative to the original first120 figure so
# the late-epoch (60+) separation between the five variants is visually
# stretched out. Values above the cap are rendered as triangles pinned to the
# top edge so the high early-training tail does not pull the limits up.
_FC2_EPE_YLIM = (2.9, 3.8)
_SINTEL_EPE_YLIM = (5.3, 6.8)

# We plot best_epe / best_sintel_epe (running-minimum curves) instead of the
# raw per-eval EPE. The raw curves have ~0.1 EPE of step-to-step noise that
# visually overlaps the genuine between-variant gaps; the running minimum is
# monotone non-increasing and exposes converged-state differences as clean
# horizontal offsets.
_FC2_METRIC = "best_epe"
_SINTEL_METRIC = "best_sintel_epe"


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUTS = _REPO_ROOT / "outputs" / "ablation_v1_fc2"
_FIG_STEM = _OUTPUTS / "ablation_v1_fc2_first130_epe_curves"

# Uniform endpoint across all five variants. A0 reached 160 ep, A1 135, A2 145,
# A4 150, A3 145, so 130 is the largest epoch every variant has at least one
# eval sample at (eval cadence = 5 epoch).
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
        label="A0 Deconv",
        csv_path=_OUTPUTS
        / "ablation_v1_fc2_run1_p100_edgeflownet_deconv"
        / "model_edgeflownet_deconv"
        / "eval_history.csv",
        color="#3b75af",
        linestyle="-",
        marker="o",
    ),
    VariantSpec(
        label="A1 Bilinear",
        csv_path=_OUTPUTS
        / "ablation_v1_fc2_run1_p100_edgeflownet_bilinear"
        / "model_edgeflownet_bilinear"
        / "eval_history.csv",
        color="#ef8636",
        linestyle="-",
        marker="o",
    ),
    VariantSpec(
        label="A2 Bilinear + ECA",
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
        label="A4 Bilinear + Gate",
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
        label="A3 Bilinear + ECA + Gate",
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
    fc2_epe: List[float] = []
    fc2_best_epe: List[float] = []
    sintel_epe: List[float] = []
    sintel_best: List[float] = []
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
            fc2_epe.append(float(row.get("epe") or "nan"))
            fc2_best_epe.append(float(row.get("best_epe") or "nan"))
            sintel_epe.append(float(row.get("sintel_epe") or "nan"))
            sintel_best.append(float(row.get("best_sintel_epe") or "nan"))
    return {
        "epoch": epochs,
        "epe": fc2_epe,
        "best_epe": fc2_best_epe,
        "sintel_epe": sintel_epe,
        "best_sintel_epe": sintel_best,
    }


def _print_summary(records: Dict[str, Dict[str, List[float]]]) -> None:
    print("=" * 88)
    print(f"{'variant':<48} {'last_ep':>7} {'fc2_epe':>9} {'fc2_best':>9} {'sin_epe':>8} {'sin_best':>9}")
    print("-" * 88)
    for spec in VARIANTS:
        h = records[spec.label]
        if not h["epoch"]:
            continue
        print(
            f"{spec.label:<48} {int(h['epoch'][-1]):>7d} "
            f"{h['epe'][-1]:>9.4f} {h['best_epe'][-1]:>9.4f} "
            f"{h['sintel_epe'][-1]:>8.3f} {h['best_sintel_epe'][-1]:>9.3f}"
        )
    print("=" * 88)


def _configure_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelweight": "bold",
            "legend.fontsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "grid.color": "#e2e6ea",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _draw_series(ax, xs, ys, ylim, spec: "VariantSpec") -> None:
    """Plot a series and pin over-cap points as triangles at the top edge.

    Mirrors the visual idiom used by the original first120 paper figure:
    values above ``ylim[1]`` are not drawn at their true value (which would
    expand the y-range and compress the late-epoch differences); instead a
    small upward-pointing triangle is rendered at the top of the panel at the
    same x position.
    """
    if not xs:
        return
    low, high = ylim
    in_x = [x for x, y in zip(xs, ys) if y <= high and y == y]  # not NaN
    in_y = [y for y in ys if y <= high and y == y]
    over_x = [x for x, y in zip(xs, ys) if y == y and y > high]
    ax.plot(
        in_x,
        in_y,
        label=spec.label,
        color=spec.color,
        linestyle=spec.linestyle,
        marker=spec.marker,
        markersize=4.5,
        linewidth=1.6,
        alpha=0.95,
    )
    if over_x:
        ax.plot(
            over_x,
            [high] * len(over_x),
            linestyle="none",
            marker="^",
            markersize=7,
            color=spec.color,
            alpha=0.9,
            clip_on=False,
        )


def _plot(records: Dict[str, Dict[str, List[float]]]) -> Tuple[Path, Path, Path]:
    _configure_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    ax_fc2, ax_sin = axes

    for spec in VARIANTS:
        h = records[spec.label]
        _draw_series(ax_fc2, h["epoch"], h[_FC2_METRIC], _FC2_EPE_YLIM, spec)
        _draw_series(ax_sin, h["epoch"], h[_SINTEL_METRIC], _SINTEL_EPE_YLIM, spec)

    ax_fc2.set_title("Best Mean EPE (running min)")
    ax_fc2.set_xlabel("Epoch")
    ax_fc2.set_ylabel("Best Mean EPE")
    ax_fc2.set_ylim(*_FC2_EPE_YLIM)
    ax_fc2.set_xlim(0, _MAX_EPOCH)

    ax_sin.set_title("Best Sintel EPE (running min)")
    ax_sin.set_xlabel("Epoch")
    ax_sin.set_ylabel("Best Sintel EPE")
    ax_sin.set_ylim(*_SINTEL_EPE_YLIM)
    ax_sin.set_xlim(0, _MAX_EPOCH)

    for ax in (ax_fc2, ax_sin):
        ax.tick_params(axis="both", which="both", length=0)

    handles, labels = ax_fc2.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))

    svg_path = _FIG_STEM.with_suffix(".svg")
    pdf_path = _FIG_STEM.with_name(_FIG_STEM.name + "_paper").with_suffix(".pdf")
    png_path = _FIG_STEM.with_name(_FIG_STEM.name + "_paper").with_suffix(".png")
    fig.savefig(svg_path)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    return svg_path, pdf_path, png_path


def main() -> int:
    matplotlib.use("Agg")
    records = {spec.label: _read_history(spec.csv_path) for spec in VARIANTS}
    for spec in VARIANTS:
        if not records[spec.label]["epoch"]:
            print(f"WARNING: empty/missing history for {spec.label}: {spec.csv_path}")
    _print_summary(records)
    svg, pdf, png = _plot(records)
    print(f"wrote: {svg}")
    print(f"wrote: {pdf}")
    print(f"wrote: {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
