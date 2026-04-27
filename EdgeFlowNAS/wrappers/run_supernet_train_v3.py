"""Supernet V3 training CLI wrapper."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from efnas.app.train_supernet_app_v3 import run_supernet_app_v3
from wrappers.run_supernet_train import _build_overrides, _put_override


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="run supernet v3 train locally without docker")
    parser.add_argument("--config", default="configs/supernet_v3_fc2_172x224.yaml", help="path to supernet v3 config yaml")

    parser.add_argument("--gpu_device", type=int, default=None, help="GPU index, set -1 for CPU")
    parser.add_argument("--gpu_devices", default=None, help="comma-separated GPU indices for multi-GPU modes")
    parser.add_argument("--multi_gpu_mode", default=None, help="single_gpu|arch_parallel")
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
    parser.add_argument("--fc2_num_workers", type=int, default=None, help="FC2 train loader workers")
    parser.add_argument("--fc2_eval_num_workers", type=int, default=None, help="FC2 eval loader workers")
    parser.add_argument("--prefetch_batches", type=int, default=None, help="bounded async prefetch depth")
    parser.add_argument("--eval_every_epoch", type=int, default=None, help="override eval frequency in epochs")
    parser.add_argument("--distill_enabled", action="store_true", help="enable distillation loss")
    parser.add_argument("--distill_lambda", type=float, default=None, help="distillation loss weight")
    parser.add_argument("--distill_teacher_type", default=None, help="distillation teacher type: edgeflownet|supernet")
    parser.add_argument("--distill_teacher_ckpt", default=None, help="teacher checkpoint prefix or path")
    parser.add_argument("--distill_teacher_arch_code", default=None, help="teacher arch code, e.g. '0 0 0 0 0 0 1 1 1 1 1'")
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


def _build_overrides_v3(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert V3 CLI arguments to config overrides."""
    overrides: Dict[str, Any] = _build_overrides(args)
    _put_override(overrides, "train.gpu_devices", args.gpu_devices)
    _put_override(overrides, "train.multi_gpu_mode", args.multi_gpu_mode)
    _put_override(overrides, "data.fc2_num_workers", args.fc2_num_workers)
    _put_override(overrides, "data.fc2_eval_num_workers", args.fc2_eval_num_workers)
    _put_override(overrides, "data.prefetch_batches", args.prefetch_batches)
    return overrides


def main() -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    overrides: Dict[str, Any] = _build_overrides_v3(args)

    result = run_supernet_app_v3(config_path=args.config, overrides=overrides, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(f"Supernet V3 training finished with exit_code={result.get('exit_code', 1)}")
    return int(result.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
