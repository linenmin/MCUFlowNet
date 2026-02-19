"""Render subnet distribution plots from analysis records."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from code.utils.json_io import write_json

DEFAULT_PLOTS = ("epe_hist", "fps_hist", "sram_hist", "epe_rank", "epe_vs_fps")


def _parse_optional_float(value: Any) -> Optional[float]:
    """Parse CSV numeric value to float."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
    except Exception:
        return None
    if np.isfinite(number):
        return float(number)
    return None


def _read_records(path: Path) -> List[Dict[str, Any]]:
    """Read analysis records CSV."""
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                {
                    "sample_index": int(row.get("sample_index", 0)),
                    "arch_code": str(row.get("arch_code", "")),
                    "epe": _parse_optional_float(row.get("epe")),
                    "fps": _parse_optional_float(row.get("fps")),
                    "sram_peak_mb": _parse_optional_float(row.get("sram_peak_mb")),
                    "vela_status": str(row.get("vela_status", "")),
                }
            )
    return records


def _parse_plot_list(text: str) -> List[str]:
    """Parse and validate plot list."""
    if not text.strip():
        return []
    names = [item.strip().lower() for item in text.split(",") if item.strip()]
    valid = {"epe_hist", "fps_hist", "sram_hist", "epe_rank", "epe_vs_fps"}
    unknown = [item for item in names if item not in valid]
    if unknown:
        raise ValueError(f"unknown plot names: {unknown}; valid={sorted(valid)}")
    out: List[str] = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


def _render_plots(records: Sequence[Dict[str, Any]], plot_names: Sequence[str], output_dir: Path, bins: int, dpi: int) -> List[str]:
    """Render PNG plots from records."""
    if not plot_names:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    generated: List[str] = []
    epe_values = [float(item["epe"]) for item in records if item.get("epe") is not None]
    fps_values = [float(item["fps"]) for item in records if item.get("fps") is not None]
    sram_values = [float(item["sram_peak_mb"]) for item in records if item.get("sram_peak_mb") is not None]

    if "epe_hist" in plot_names and epe_values:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(np.asarray(epe_values, dtype=np.float64), bins=max(5, int(bins)), alpha=0.85, edgecolor="black")
        ax.set_title("Subnet EPE Distribution")
        ax.set_xlabel("EPE")
        ax.set_ylabel("Count")
        path = output_dir / "epe_hist.png"
        fig.tight_layout()
        fig.savefig(path, dpi=int(dpi))
        plt.close(fig)
        generated.append(path.name)

    if "fps_hist" in plot_names and fps_values:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(np.asarray(fps_values, dtype=np.float64), bins=max(5, int(bins)), alpha=0.85, edgecolor="black")
        ax.set_title("Subnet FPS Distribution")
        ax.set_xlabel("FPS")
        ax.set_ylabel("Count")
        path = output_dir / "fps_hist.png"
        fig.tight_layout()
        fig.savefig(path, dpi=int(dpi))
        plt.close(fig)
        generated.append(path.name)

    if "sram_hist" in plot_names and sram_values:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(np.asarray(sram_values, dtype=np.float64), bins=max(5, int(bins)), alpha=0.85, edgecolor="black")
        ax.set_title("Subnet SRAM Peak Distribution")
        ax.set_xlabel("SRAM Peak (MB)")
        ax.set_ylabel("Count")
        path = output_dir / "sram_hist.png"
        fig.tight_layout()
        fig.savefig(path, dpi=int(dpi))
        plt.close(fig)
        generated.append(path.name)

    if "epe_rank" in plot_names and epe_values:
        sorted_epe = np.sort(np.asarray(epe_values, dtype=np.float64))
        ranks = np.arange(1, len(sorted_epe) + 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ranks, sorted_epe, linewidth=2.0)
        ax.set_title("Subnet EPE Rank Curve (Lower is Better)")
        ax.set_xlabel("Rank")
        ax.set_ylabel("EPE")
        path = output_dir / "epe_rank_curve.png"
        fig.tight_layout()
        fig.savefig(path, dpi=int(dpi))
        plt.close(fig)
        generated.append(path.name)

    if "epe_vs_fps" in plot_names:
        pairs = [
            (float(item["fps"]), float(item["epe"]))
            for item in records
            if item.get("fps") is not None and item.get("epe") is not None
        ]
        if pairs:
            x = np.asarray([item[0] for item in pairs], dtype=np.float64)
            y = np.asarray([item[1] for item in pairs], dtype=np.float64)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(x, y, s=18, alpha=0.75)
            ax.set_title("EPE vs FPS")
            ax.set_xlabel("FPS")
            ax.set_ylabel("EPE")
            path = output_dir / "epe_vs_fps_scatter.png"
            fig.tight_layout()
            fig.savefig(path, dpi=int(dpi))
            plt.close(fig)
            generated.append(path.name)

    return generated


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="render subnet profile plots from analysis records")
    parser.add_argument("--analysis_dir", required=True, help="analysis folder that contains records.csv")
    parser.add_argument("--output_dir", default=None, help="optional output folder for png plots")
    parser.add_argument("--plots", default=",".join(DEFAULT_PLOTS), help="plot list: epe_hist,fps_hist,sram_hist,epe_rank,epe_vs_fps")
    parser.add_argument("--hist_bins", type=int, default=30, help="histogram bins")
    parser.add_argument("--dpi", type=int, default=150, help="png dpi")
    return parser


def main() -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    records_path = analysis_dir / "records.csv"
    if not records_path.exists():
        raise FileNotFoundError(f"records.csv not found: {records_path}")

    output_dir = Path(args.output_dir) if args.output_dir else (analysis_dir.parent / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _read_records(path=records_path)
    plot_names = _parse_plot_list(text=str(args.plots))
    generated = _render_plots(
        records=records,
        plot_names=plot_names,
        output_dir=output_dir,
        bins=int(args.hist_bins),
        dpi=int(args.dpi),
    )

    manifest_path = output_dir / "plot_manifest.json"
    payload = {
        "status": "ok",
        "analysis_dir": str(analysis_dir),
        "records_csv": str(records_path),
        "output_dir": str(output_dir),
        "plots_requested": plot_names,
        "plots_generated": generated,
        "num_records": int(len(records)),
    }
    write_json(str(manifest_path), payload)

    result = {
        "status": "ok",
        "output_dir": str(output_dir),
        "plot_manifest": str(manifest_path),
        "plots_generated": generated,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
