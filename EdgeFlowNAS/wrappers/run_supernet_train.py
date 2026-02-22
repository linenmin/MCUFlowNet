"""Supernet training CLI wrapper."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.app.train_supernet_app import run_supernet_app


def _put_override(overrides: Dict[str, Any], key: str, value: Any) -> None:
    """Set override only when value is not None."""
    if value is None:
        return
    overrides[key] = value


def _build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert CLI arguments to nested config overrides."""
    overrides: Dict[str, Any] = {}
    _put_override(overrides, "train.gpu_device", args.gpu_device)
    _put_override(overrides, "train.num_epochs", args.num_epochs)
    _put_override(overrides, "train.steps_per_epoch", args.steps_per_epoch)
    _put_override(overrides, "train.batch_size", args.batch_size)
    _put_override(overrides, "train.micro_batch_size", args.micro_batch_size)
    _put_override(overrides, "train.lr", args.lr)
    _put_override(overrides, "train.supernet_mode", args.supernet_mode)

    _put_override(overrides, "runtime.experiment_name", args.experiment_name)

    _put_override(overrides, "data.dataset", args.dataset)
    _put_override(overrides, "data.base_path", args.base_path)
    _put_override(overrides, "data.train_dir", args.train_dir)
    _put_override(overrides, "data.val_dir", args.val_dir)
    _put_override(overrides, "eval.eval_every_epoch", args.eval_every_epoch)
    _put_override(overrides, "train.distill.lambda", args.distill_lambda)
    _put_override(overrides, "train.distill.teacher_ckpt", args.distill_teacher_ckpt)
    _put_override(overrides, "train.distill.teacher_arch_code", args.distill_teacher_arch_code)
    _put_override(overrides, "train.distill.layer_weights", args.distill_layer_weights)

    _put_override(overrides, "checkpoint.resume_experiment_name", args.resume_experiment_name)
    if args.load_checkpoint:
        _put_override(overrides, "checkpoint.load_checkpoint", True)
    if args.reset_early_stop_on_resume:
        _put_override(overrides, "checkpoint.reset_early_stop_on_resume", True)
    if args.distill_enabled:
        _put_override(overrides, "train.distill.enabled", True)
    return overrides


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="run supernet train locally without docker")
    parser.add_argument("--config", default="configs/supernet_fc2_180x240.yaml", help="path to supernet config yaml")

    parser.add_argument("--gpu_device", type=int, default=None, help="GPU index, set -1 for CPU")
    parser.add_argument("--num_epochs", type=int, default=None, help="number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="steps per epoch")
    parser.add_argument("--batch_size", type=int, default=None, help="mini-batch size")
    parser.add_argument("--micro_batch_size", type=int, default=None, help="micro batch size for gradient accumulation")
    parser.add_argument("--lr", type=float, default=None, help="learning rate")
    parser.add_argument("--supernet_mode", default=None, help="supernet mode name")

    parser.add_argument("--dataset", default=None, help="dataset name, e.g. FC2")
    parser.add_argument("--base_path", default=None, help="dataset root path")
    parser.add_argument("--train_dir", default=None, help="train folder path, absolute or relative to base_path")
    parser.add_argument("--val_dir", default=None, help="val folder path, absolute or relative to base_path")
    # 允许在断点恢复时从 CLI 覆盖评估频率。
    parser.add_argument("--eval_every_epoch", type=int, default=None, help="override eval frequency in epochs")
    parser.add_argument("--distill_enabled", action="store_true", help="enable distillation loss")
    parser.add_argument("--distill_lambda", type=float, default=None, help="distillation loss weight")
    parser.add_argument("--distill_teacher_ckpt", default=None, help="teacher checkpoint prefix or path")
    parser.add_argument("--distill_teacher_arch_code", default=None, help="teacher arch code, e.g. '0 0 0 0 2 1 2 2 2'")
    parser.add_argument("--distill_layer_weights", default=None, help="distill layer weights, e.g. '1.0 1.0 1.0'")

    parser.add_argument("--experiment_name", default=None, help="experiment name under outputs")
    parser.add_argument("--resume_experiment_name", default=None, help="resume source experiment name")
    parser.add_argument("--load_checkpoint", action="store_true", help="resume from checkpoint")
    parser.add_argument(
        "--reset_early_stop_on_resume",
        action="store_true",
        help="reset early-stop counters when resuming",
    )

    parser.add_argument("--dry_run", action="store_true", help="print merged config and exit")
    return parser


def main() -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    overrides = _build_overrides(args)

    result = run_supernet_app(config_path=args.config, overrides=overrides, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(f"Supernet training finished with exit_code={result.get('exit_code', 1)}")
    return int(result.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
