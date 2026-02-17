"""Supernet evaluation CLI wrapper."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _append_opt(cmd, name: str, value) -> None:
    """Append option to command list when provided."""
    if value is None:
        return
    cmd.extend([name, str(value)])


def _build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(description="run supernet eval without editing yaml")
    parser.add_argument("--config", default="configs/supernet_fc2_180x240.yaml", help="path to supernet config yaml")
    parser.add_argument("--checkpoint_type", choices=["best", "last"], default="last", help="checkpoint type")
    parser.add_argument("--batch_size", type=int, default=None, help="eval batch size")
    parser.add_argument("--bn_recal_batches", type=int, default=None, help="bn recalibration batches")
    parser.add_argument("--num_workers", type=int, default=None, help="parallel worker processes for arch eval")

    parser.add_argument("--experiment_name", default=None, help="override runtime.experiment_name")
    parser.add_argument("--base_path", default=None, help="override data.base_path")
    parser.add_argument("--train_dir", default=None, help="override data.train_dir")
    parser.add_argument("--val_dir", default=None, help="override data.val_dir")
    parser.add_argument("--train_batch_size", type=int, default=None, help="override train.batch_size")
    parser.add_argument("--seed", type=int, default=None, help="override runtime.seed")

    parser.add_argument("--cpu_only", action="store_true", help="force CPU-only eval")
    parser.add_argument("--dry_run", action="store_true", help="print eval command and exit")
    return parser


def main() -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "code.nas.supernet_eval",
        "--config",
        args.config,
        "--eval_only",
        "--checkpoint_type",
        args.checkpoint_type,
    ]
    _append_opt(cmd, "--batch_size", args.batch_size)
    _append_opt(cmd, "--bn_recal_batches", args.bn_recal_batches)
    _append_opt(cmd, "--num_workers", args.num_workers)
    _append_opt(cmd, "--experiment_name", args.experiment_name)
    _append_opt(cmd, "--base_path", args.base_path)
    _append_opt(cmd, "--train_dir", args.train_dir)
    _append_opt(cmd, "--val_dir", args.val_dir)
    _append_opt(cmd, "--train_batch_size", args.train_batch_size)
    _append_opt(cmd, "--seed", args.seed)
    if args.cpu_only:
        cmd.append("--cpu_only")

    if args.dry_run:
        print(" ".join(shlex.quote(part) for part in cmd))
        return 0

    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
