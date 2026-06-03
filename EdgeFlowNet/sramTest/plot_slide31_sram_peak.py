#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create the Slide 3.1 SRAM-peak figure from a Vela per-layer CSV.

The plot deliberately merges low-level Vela operators into architecture-level
stages. For SRAM, the right aggregation is max-within-stage: the slide explains
which stage creates the peak memory requirement, not total memory over time.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CSV = Path(__file__).resolve().parent / "output" / "sram_test_modified_per-layer.csv"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "figures"


@dataclass(frozen=True)
class VelaRow:
    index: int
    op: str
    name: str
    sram_bytes: float
    peak_pct: float


@dataclass(frozen=True)
class Stage:
    label: str
    detail: str
    rows: tuple[int, ...]
    kind: str


ORIGINAL_STAGES = (
    Stage("Stem\nconvs", "Initial 7x7 and 5x5 Conv-BN-ReLU", (1, 2), "conv"),
    Stage("Encoder\nResBlocks", "Repeated encoder residual blocks", (3, 4, 5, 7, 8, 9), "conv"),
    Stage("Encoder\ndownsample", "Encoder transition convolutions", (6, 10), "conv"),
    Stage("Decoder\nResBlocks", "Repeated transposed-conv residual blocks", (11, 12, 13, 15, 16, 17), "tconv"),
    Stage("Decoder\nupsample", "Intermediate transposed-conv upsampling", (14, 18), "tconv"),
    Stage("Multi-scale\nflow heads", "Transposed-conv flow heads", (19, 20, 21, 22, 23), "peak"),
    Stage("Output\nresize/add", "Multi-scale output accumulation", (24, 25, 26, 27, 28), "output"),
)

BILINEAR_STAGES = (
    Stage("Stem\nconvs", "Initial 7x7 and 5x5 Conv-BN-ReLU", (1, 2), "conv"),
    Stage("Encoder\nResBlocks", "Repeated encoder residual blocks", (3, 4, 5, 7, 8, 9), "conv"),
    Stage("Encoder\ndownsample", "Encoder transition convolutions", (6, 10), "conv"),
    Stage("Decoder\nResBlocks", "Repeated decoder residual blocks", (11, 12, 13, 16, 17, 18), "tconv"),
    Stage("Decoder\nupsample", "Bilinear resize plus following convolution", (14, 15, 19, 20), "tconv"),
    Stage("Multi-scale\nflow heads", "Bilinear-resized flow heads", (21, 22, 23, 24, 25, 26, 27), "peak"),
    Stage("Output\nresize/add", "Multi-scale output accumulation", (28, 29, 30, 31), "output"),
)

ABLATION_A3_STAGES = (
    Stage("Stem\nconvs", "Initial 7x7 and 5x5 Conv-BN-ReLU", (1, 2), "conv"),
    Stage("Encoder\nResBlocks", "Repeated encoder residual blocks", (3, 4, 5, 7, 8, 9), "conv"),
    Stage("Encoder\ndownsample", "Encoder transition convolutions", (6, 10), "conv"),
    Stage("Decoder\nResBlocks", "Repeated decoder residual blocks", (11, 12, 13, 16, 17, 18), "tconv"),
    Stage("Decoder\nupsample", "Bilinear resize plus following convolution", (14, 15, 19, 20), "tconv"),
    Stage(
        "Multi-scale\nflow heads",
        "Bilinear-resized flow heads with ECA and 1/4 global-gate side path merged into the affected stage",
        (21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38),
        "peak",
    ),
    Stage("Output\nresize/add", "Multi-scale output accumulation", (39, 40, 41, 42, 43), "output"),
)

FIXED_V3_STAGES = (
    Stage("Stem\nconvs", "Selected stem convolutions", (1, 2), "conv"),
    Stage("Encoder\nResBlocks", "Selected encoder residual blocks", (3, 4, 5, 7, 8, 9), "conv"),
    Stage("Encoder\ndownsample", "Encoder downsampling transitions", (6, 10), "conv"),
    Stage("Bottleneck\nblock + ECA", "DB0 bottleneck block plus serial bottleneck ECA", (11, 12, 13, 14, 15, 16, 17, 18, 19), "eca"),
    Stage("Decoder / gate\nregion", "Up1, DB1, Up2 and 1/4 global gate", (24, 25, 26, 27, 28, 29, 30, 31), "tconv"),
    Stage("Multi-scale\nflow heads", "Bilinear-resized flow heads", (32, 33, 34, 35, 36, 37, 38), "peak"),
    Stage("Output\nresize/add", "Multi-scale output accumulation", (39, 40, 41, 42, 43), "output"),
)


def read_vela_rows(csv_path: Path) -> list[VelaRow]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader, start=1):
            rows.append(
                VelaRow(
                    index=i,
                    op=row["TFLite_operator"].strip(),
                    name=row["Name"].split(";")[0].strip(),
                    sram_bytes=float(row["SRAM Usage"]),
                    peak_pct=float(row["Peak%"]),
                )
            )
    return rows


def stage_values(rows: list[VelaRow], variant: str) -> list[dict[str, object]]:
    if variant == "fixed_v3":
        stages = FIXED_V3_STAGES
    elif variant == "ablation_a3":
        stages = ABLATION_A3_STAGES
    elif variant == "bilinear":
        stages = BILINEAR_STAGES
    else:
        stages = ORIGINAL_STAGES
    by_index = {r.index: r for r in rows}
    out = []
    for stage in stages:
        members = [by_index[i] for i in stage.rows if i in by_index]
        if not members:
            continue
        peak_row = max(members, key=lambda r: r.sram_bytes)
        out.append(
            {
                "label": stage.label,
                "detail": stage.detail,
                "kind": stage.kind,
                "sram_mb": peak_row.sram_bytes / (1024 * 1024),
                "peak_pct": peak_row.peak_pct,
                "peak_op": peak_row.op,
                "peak_row": peak_row.index,
                "peak_name": peak_row.name,
                "merged_rows": ",".join(str(i) for i in stage.rows),
            }
        )
    return out


def write_grouped_csv(values: list[dict[str, object]], path: Path) -> None:
    fields = [
        "label",
        "detail",
        "kind",
        "sram_mb",
        "peak_pct",
        "peak_op",
        "peak_row",
        "merged_rows",
        "peak_name",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in values:
            clean = dict(row)
            clean["label"] = str(clean["label"]).replace("\n", " ")
            writer.writerow(clean)


def side_path_values(rows: list[VelaRow], variant: str) -> list[dict[str, object]]:
    if variant == "fixed_v3":
        wanted = (
            ("Global gate", "global_gate_4x_scale"),
        )
    elif variant == "ablation_a3":
        wanted = (
            ("ECA scale", "eca_bottleneck_scale"),
            ("Global gate", "global_gate_4x_scale"),
        )
    else:
        return []
    out = []
    for label, needle in wanted:
        matches = [r for r in rows if needle in r.name]
        if not matches:
            continue
        peak_row = max(matches, key=lambda r: r.sram_bytes)
        out.append(
            {
                "label": label,
                "sram_mb": peak_row.sram_bytes / (1024 * 1024),
                "peak_row": peak_row.index,
                "peak_op": peak_row.op,
                "peak_name": peak_row.name,
            }
        )
    return out


def draw(
    values: list[dict[str, object]],
    out_dir: Path,
    stem: str,
    side_path: list[dict[str, object]] | None = None,
    side_path_panel: bool = False,
    stack_gate_footprint: bool = False,
    fig_width: float | None = None,
    fig_height: float | None = None,
    title: str = "SRAM Profile",
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    side_path = side_path or []

    labels = [str(v["label"]).replace("\n", " ") for v in values]
    sram = np.array([float(v["sram_mb"]) for v in values])
    y = np.arange(len(values))
    peak_i = int(np.argmax(sram))

    colors = []
    for v in values:
        if v["kind"] == "peak":
            colors.append("#D84A3A")
        elif v["kind"] == "eca":
            colors.append("#8DC58F")
        elif v["kind"] == "tconv":
            colors.append("#F2B36D")
        elif v["kind"] == "output":
            colors.append("#9AA4AE")
        else:
            colors.append("#89B7D8")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 21,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 16,
            "figure.dpi": 120,
        }
    )

    if side_path_panel and side_path:
        fig = plt.figure(figsize=(fig_width or 11.0, fig_height or 10.3), constrained_layout=True)
        gs = fig.add_gridspec(
            3,
            2,
            width_ratios=[4.8, 2.0],
            height_ratios=[1.2, 1.0, 1.2],
            wspace=0.12,
            hspace=0.08,
        )
        ax = fig.add_subplot(gs[:, 0])
        side_ax = fig.add_subplot(gs[1, 1])
    else:
        fig, ax = plt.subplots(figsize=(fig_width or 8.1, fig_height or 10.3), constrained_layout=True)
        side_ax = None

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.barh(y, sram, height=0.68, color=colors, edgecolor="#2E3440", linewidth=1.3)
    bars[peak_i].set_edgecolor("#8F1F16")
    bars[peak_i].set_linewidth(3.2)

    stacked_i = None
    stacked_extra = []
    if stack_gate_footprint and side_path:
        for idx, value in enumerate(values):
            clean_label = str(value["label"]).replace("\n", " ")
            if clean_label in ("Decoder upsample", "Decoder / gate region"):
                stacked_i = idx
                break
        if stacked_i is not None:
            stacked_extra = [(str(v["label"]), float(v["sram_mb"])) for v in side_path]
            left = float(sram[stacked_i])
            stack_colors = {
                "ECA scale": "#8DC58F",
                "Global gate": "#3F9B73",
            }
            stack_text_colors = {
                "ECA scale": "#1F3D32",
                "Global gate": "#F4FBF7",
            }
            for label, width in stacked_extra:
                ax.barh(
                    y[stacked_i],
                    width,
                    left=left,
                    height=0.68,
                    color=stack_colors.get(label, "#7CB68A"),
                    edgecolor="#2E3440",
                    linewidth=1.3,
                    label=label,
                )
                if width >= 0.12:
                    ax.text(
                        left + width / 2,
                        y[stacked_i],
                        f"+{width:.2f}",
                        ha="center",
                        va="center",
                        fontsize=13,
                        color=stack_text_colors.get(label, "#1F3D32"),
                        fontweight="bold",
                    )
                left += width
            labels[stacked_i] = "Decoder / gate region"

    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    if stack_gate_footprint:
        ax.set_xlabel("SRAM footprint (MB)")
    else:
        ax.set_xlabel("Peak SRAM (MB)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    visual_max = max(sram)
    if stacked_i is not None and stacked_extra:
        visual_max = max(visual_max, float(sram[stacked_i]) + sum(width for _, width in stacked_extra))
    ax.set_xlim(0, visual_max * 1.45)
    ax.grid(axis="x", color="#D6DCE2", linewidth=1.0, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6B7280")
    ax.spines["bottom"].set_color("#6B7280")

    for i, (bar, value) in enumerate(zip(bars, sram)):
        if i == peak_i:
            continue
        if stack_gate_footprint and stacked_i is not None and i == stacked_i:
            total_value = value + sum(width for _, width in stacked_extra)
            ax.text(
                total_value + visual_max * 0.07,
                bar.get_y() + bar.get_height() / 2,
                f"{total_value:.2f}",
                ha="left",
                va="center",
                fontsize=14,
                color="#2F5D4A",
                fontweight="bold",
            )
            continue
        ax.text(
            value + max(sram) * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            ha="left",
            va="center",
            fontsize=14,
            color="#3E4651",
            fontweight="bold",
        )

    peak_value = sram[peak_i]
    ax.text(
        peak_value + max(sram) * 0.04,
        y[peak_i],
        f"{peak_value:.2f} MB",
        ha="left",
        va="center",
        fontsize=18,
        color="#8F1F16",
        fontweight="bold",
    )

    ax.tick_params(axis="y", pad=10)

    if stack_gate_footprint and stacked_extra:
        from matplotlib.patches import Patch

        handles = [Patch(facecolor="#F2B36D", edgecolor="#2E3440", label="Bilinear stage peak")]
        if any(label == "ECA scale" for label, _ in stacked_extra):
            handles.append(Patch(facecolor="#8DC58F", edgecolor="#2E3440", label="ECA local"))
        if any(label == "Global gate" for label, _ in stacked_extra):
            handles.append(Patch(facecolor="#3F9B73", edgecolor="#2E3440", label="Global gate local"))
        ax.legend(
            handles=handles,
            loc="upper right",
            frameon=False,
            fontsize=13,
            handlelength=1.4,
            borderpad=0.2,
            labelspacing=0.5,
        )

    if side_ax is not None:
        side_labels = [str(v["label"]) for v in side_path]
        side_sram = np.array([float(v["sram_mb"]) for v in side_path])
        side_y = np.arange(len(side_path))
        side_colors = ["#7CB68A", "#3F9B73"]

        side_ax.set_facecolor("white")
        side_bars = side_ax.barh(
            side_y,
            side_sram,
            height=0.52,
            color=side_colors[: len(side_path)],
            edgecolor="#2E3440",
            linewidth=1.2,
            hatch="//",
        )
        side_ax.set_title("Parallel gate path", loc="left", pad=12, fontweight="bold")
        side_ax.set_xlabel("Local SRAM (MB)")
        side_ax.set_yticks(side_y)
        side_ax.set_yticklabels(side_labels)
        side_ax.invert_yaxis()
        side_ax.set_xlim(0, max(float(max(side_sram)) * 1.55, 0.62))
        side_ax.grid(axis="x", color="#D6DCE2", linewidth=1.0, alpha=0.9)
        side_ax.set_axisbelow(True)
        side_ax.spines["top"].set_visible(False)
        side_ax.spines["right"].set_visible(False)
        side_ax.spines["left"].set_color("#6B7280")
        side_ax.spines["bottom"].set_color("#6B7280")
        side_ax.tick_params(axis="y", pad=8)

        for bar, value in zip(side_bars, side_sram):
            side_ax.text(
                value + max(float(max(side_sram)) * 0.045, 0.015),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f} MB",
                ha="left",
                va="center",
                fontsize=15,
                color="#2F5D4A",
                fontweight="bold",
            )

    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw Slide 3.1 SRAM peak figure from a Vela per-layer CSV.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Vela *_per-layer.csv path.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output figure directory.")
    parser.add_argument("--stem", default="slide31_sram_peak", help="Output file stem.")
    parser.add_argument("--variant", choices=("original", "bilinear", "ablation_a3", "fixed_v3"), default="original", help="Stage grouping preset.")
    parser.add_argument("--side-path-panel", action="store_true", help="Show local SRAM for inserted parallel side-path ops.")
    parser.add_argument("--fig-width", type=float, default=None, help="Output figure width in inches.")
    parser.add_argument("--fig-height", type=float, default=None, help="Output figure height in inches.")
    parser.add_argument("--title", default="SRAM Profile", help="Figure title.")
    parser.add_argument(
        "--stack-gate-footprint",
        action="store_true",
        help="Draw inserted side-path local footprints as a hatched extension on the decoder/gate region.",
    )
    args = parser.parse_args()

    rows = read_vela_rows(args.csv)
    values = stage_values(rows, args.variant)
    side_path = side_path_values(rows, args.variant) if (args.side_path_panel or args.stack_gate_footprint) else []
    grouped_csv = args.out_dir / f"{args.stem}_grouped.csv"
    write_grouped_csv(values, grouped_csv)
    png, pdf = draw(
        values,
        args.out_dir,
        args.stem,
        side_path=side_path,
        side_path_panel=args.side_path_panel,
        stack_gate_footprint=args.stack_gate_footprint,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        title=args.title,
    )

    print(f"Wrote: {png}")
    print(f"Wrote: {pdf}")
    print(f"Wrote: {grouped_csv}")


if __name__ == "__main__":
    main()
