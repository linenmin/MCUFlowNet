"""Supernet subnet-distribution analysis CLI wrapper."""

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
    """Build command-line parser."""
    parser = argparse.ArgumentParser(description="run subnet distribution analysis without editing yaml")
    parser.add_argument("--config", default="configs/supernet_fc2_180x240.yaml", help="path to supernet config yaml")
    parser.add_argument("--checkpoint_type", choices=["best", "last"], default="best", help="checkpoint type")
    parser.add_argument("--num_arch_samples", type=int, default=None, help="number of sampled subnets")
    parser.add_argument("--sample_seed", type=int, default=None, help="random seed for sampling")
    parser.add_argument("--exclude_eval_pool", action="store_true", help="exclude fixed eval_pool_12 from samples")

    parser.add_argument("--bn_recal_batches", type=int, default=None, help="override bn recalibration batches")
    parser.add_argument("--eval_batches_per_arch", type=int, default=None, help="override eval batches per arch")
    parser.add_argument("--batch_size", type=int, default=None, help="eval batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="parallel worker processes")
    parser.add_argument("--cpu_only", action="store_true", help="force CPU-only eval")

    parser.add_argument("--enable_vela", action="store_true", help="enable vela benchmark")
    parser.add_argument("--vela_mode", choices=["basic", "verbose"], default=None, help="vela mode")
    parser.add_argument("--vela_optimise", choices=["Performance", "Size"], default=None, help="vela optimise mode")
    parser.add_argument("--vela_limit", type=int, default=None, help="limit vela runs")
    parser.add_argument("--vela_rep_dataset_samples", type=int, default=None, help="rep-dataset samples for int8 export")
    parser.add_argument("--vela_float32", action="store_true", help="export float32 tflite")
    parser.add_argument("--vela_keep_artifacts", action="store_true", help="keep vela temp folders")
    parser.add_argument("--vela_verbose_log", action="store_true", help="show vela detailed logs")

    parser.add_argument("--plots", default=None, help="deprecated: use run_supernet_subnet_distribution_plot.py")
    parser.add_argument("--hist_bins", type=int, default=None, help="deprecated: use run_supernet_subnet_distribution_plot.py")
    parser.add_argument("--top_k", type=int, default=None, help="top/bottom K rows in summary")
    parser.add_argument("--output_tag", default=None, help="suffix tag for output folder")

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
        "code.nas.supernet_subnet_distribution",
        "--config",
        args.config,
        "--checkpoint_type",
        args.checkpoint_type,
    ]
    _append_opt(cmd, "--num_arch_samples", args.num_arch_samples)
    _append_opt(cmd, "--sample_seed", args.sample_seed)
    if args.exclude_eval_pool:
        cmd.append("--exclude_eval_pool")

    _append_opt(cmd, "--bn_recal_batches", args.bn_recal_batches)
    _append_opt(cmd, "--eval_batches_per_arch", args.eval_batches_per_arch)
    _append_opt(cmd, "--batch_size", args.batch_size)
    _append_opt(cmd, "--num_workers", args.num_workers)
    if args.cpu_only:
        cmd.append("--cpu_only")

    if args.enable_vela:
        cmd.append("--enable_vela")
    _append_opt(cmd, "--vela_mode", args.vela_mode)
    _append_opt(cmd, "--vela_optimise", args.vela_optimise)
    _append_opt(cmd, "--vela_limit", args.vela_limit)
    _append_opt(cmd, "--vela_rep_dataset_samples", args.vela_rep_dataset_samples)
    if args.vela_float32:
        cmd.append("--vela_float32")
    if args.vela_keep_artifacts:
        cmd.append("--vela_keep_artifacts")
    if args.vela_verbose_log:
        cmd.append("--vela_verbose_log")

    _append_opt(cmd, "--plots", args.plots)
    _append_opt(cmd, "--hist_bins", args.hist_bins)
    _append_opt(cmd, "--top_k", args.top_k)
    _append_opt(cmd, "--output_tag", args.output_tag)

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
