"""Subnet-distribution plot CLI wrapper."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _append_opt(cmd, name: str, value) -> None:
    """Append option when value exists."""
    if value is None:
        return
    cmd.extend([name, str(value)])


def _build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(description="render subnet distribution png charts")
    parser.add_argument("--analysis_dir", required=True, help="analysis dir that contains records.csv")
    parser.add_argument("--output_dir", default=None, help="optional output dir for png plots")
    parser.add_argument("--plots", default=None, help="plot list: epe_hist,fps_hist,sram_hist,epe_rank,epe_vs_fps")
    parser.add_argument("--hist_bins", type=int, default=None, help="histogram bins")
    parser.add_argument("--dpi", type=int, default=None, help="png dpi")
    parser.add_argument("--dry_run", action="store_true", help="print command and exit")
    return parser


def main() -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "code.nas.supernet_subnet_distribution_plot",
        "--analysis_dir",
        args.analysis_dir,
    ]
    _append_opt(cmd, "--output_dir", args.output_dir)
    _append_opt(cmd, "--plots", args.plots)
    _append_opt(cmd, "--hist_bins", args.hist_bins)
    _append_opt(cmd, "--dpi", args.dpi)

    if args.dry_run:
        print(" ".join(shlex.quote(part) for part in cmd))
        return 0

    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
