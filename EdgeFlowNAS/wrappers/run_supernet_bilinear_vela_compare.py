"""Wrapper for supernet vs bilinear Vela comparison."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _append_opt(cmd, name: str, value) -> None:
    """Append option when value is provided."""
    if value is None:
        return
    cmd.extend([name, str(value)])


def _build_parser() -> argparse.ArgumentParser:
    """Build cli parser."""
    parser = argparse.ArgumentParser(description="run one supernet-subnet vs bilinear vela comparison")
    parser.add_argument("--config", default="configs/supernet_fc2_180x240.yaml", help="path to supernet config yaml")
    parser.add_argument("--checkpoint_type", choices=["best", "last"], default="best", help="checkpoint type")
    parser.add_argument("--skip_checkpoint", action="store_true", help="export supernet with random init")
    parser.add_argument("--arch_code", default="0,0,0,0,2,1,2,2,2", help="supernet arch code")
    parser.add_argument("--output_tag", default=None, help="suffix tag for output folder")
    parser.add_argument("--vela_mode", choices=["basic", "verbose"], default=None, help="vela mode")
    parser.add_argument("--vela_optimise", choices=["Performance", "Size"], default=None, help="vela optimise mode")
    parser.add_argument("--vela_rep_dataset_samples", type=int, default=None, help="rep-dataset samples for int8 export")
    parser.add_argument("--vela_float32", action="store_true", help="export float32 tflite")
    parser.add_argument("--vela_verbose_log", action="store_true", help="show vela detailed logs")
    parser.add_argument("--keep_tflite", action="store_true", help="keep generated tflite files")
    parser.add_argument("--experiment_name", default=None, help="override runtime.experiment_name")
    parser.add_argument("--base_path", default=None, help="override data.base_path")
    parser.add_argument("--train_dir", default=None, help="override data.train_dir")
    parser.add_argument("--val_dir", default=None, help="override data.val_dir")
    parser.add_argument("--train_batch_size", type=int, default=None, help="override train.batch_size")
    parser.add_argument("--seed", type=int, default=None, help="override runtime.seed")
    parser.add_argument("--dry_run", action="store_true", help="print command and exit")
    return parser


def main() -> int:
    """CLI entry."""
    parser = _build_parser()
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "code.nas.supernet_bilinear_vela_compare",
        "--config",
        args.config,
        "--checkpoint_type",
        args.checkpoint_type,
        "--arch_code",
        args.arch_code,
    ]
    _append_opt(cmd, "--output_tag", args.output_tag)
    _append_opt(cmd, "--vela_mode", args.vela_mode)
    _append_opt(cmd, "--vela_optimise", args.vela_optimise)
    _append_opt(cmd, "--vela_rep_dataset_samples", args.vela_rep_dataset_samples)
    if args.skip_checkpoint:
        cmd.append("--skip_checkpoint")
    if args.vela_float32:
        cmd.append("--vela_float32")
    if args.vela_verbose_log:
        cmd.append("--vela_verbose_log")
    if args.keep_tflite:
        cmd.append("--keep_tflite")
    _append_opt(cmd, "--experiment_name", args.experiment_name)
    _append_opt(cmd, "--base_path", args.base_path)
    _append_opt(cmd, "--train_dir", args.train_dir)
    _append_opt(cmd, "--val_dir", args.val_dir)
    _append_opt(cmd, "--train_batch_size", args.train_batch_size)
    _append_opt(cmd, "--seed", args.seed)

    if args.dry_run:
        print(" ".join(shlex.quote(part) for part in cmd))
        return 0

    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
