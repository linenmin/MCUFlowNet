"""CLI wrapper for the V2 inherited-weight FC2/Sintel rank-consistency diagnostic."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from efnas.nas.supernet_v2_rank_consistency import run_rank_consistency_diagnostic


def _put_override(overrides: Dict[str, Any], key: str, value: Any) -> None:
    """Set one nested override only when value is provided."""
    if value is None:
        return
    overrides[key] = value


def _build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Map selected CLI args onto config overrides."""
    overrides: Dict[str, Any] = {}
    _put_override(overrides, "data.base_path", args.base_path)
    _put_override(overrides, "data.train_dir", args.train_dir)
    _put_override(overrides, "data.val_dir", args.val_dir)
    _put_override(overrides, "train.batch_size", args.batch_size)
    _put_override(overrides, "eval.eval_batches_per_arch", args.eval_batches_per_arch)
    return overrides


def _build_parser() -> argparse.ArgumentParser:
    """Build wrapper CLI parser."""
    parser = argparse.ArgumentParser(description="compare FC2 and Sintel rankings on inherited-weight supernet_v2 subnets")
    parser.add_argument("--config", default="configs/supernet_fc2_172x224_v2.yaml", help="path to supernet V2 config yaml")
    parser.add_argument("--experiment_dir", default=None, help="trained supernet_v2 experiment dir containing checkpoints/")
    parser.add_argument("--checkpoint_type", choices=["best", "last"], default="best", help="checkpoint to evaluate")
    parser.add_argument("--num_arch_samples", type=int, default=50, help="number of V2 subnets to probe")
    parser.add_argument("--sample_seed", type=int, default=42, help="sampling seed for the probe pool")
    parser.add_argument("--output_dir", default=None, help="where to write diagnostic outputs")

    parser.add_argument("--base_path", default=None, help="override data.base_path")
    parser.add_argument("--train_dir", default=None, help="override data.train_dir")
    parser.add_argument("--val_dir", default=None, help="override data.val_dir")
    parser.add_argument("--batch_size", type=int, default=None, help="override FC2 eval batch size via config")

    parser.add_argument("--dataset_root", default=None, help="path to Sintel root (usually Datasets/Sintel)")
    parser.add_argument(
        "--sintel_list",
        default="EdgeFlowNet/code/dataset_paths/MPI_Sintel_Final_train_list.txt",
        help="path to the Sintel split list",
    )
    parser.add_argument("--sintel_patch_size", default="416,1024", help="Sintel evaluation size H,W")
    parser.add_argument("--max_sintel_samples", type=int, default=None, help="optional cap for quick pilot runs")

    parser.add_argument("--bn_recal_batches", type=int, default=16, help="FC2-train BN recalibration batches per arch")
    parser.add_argument("--bn_recal_batch_size", type=int, default=None, help="BN recalibration batch size")
    parser.add_argument("--eval_batches_per_arch", type=int, default=16, help="FC2 val batches per arch")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="FC2 val batch size")
    parser.add_argument("--gpu_device", type=int, default=None, help="GPU index to expose via CUDA_VISIBLE_DEVICES")

    parser.add_argument("--dry_run", action="store_true", help="print merged config + sample preview and exit")
    return parser


def main() -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    overrides = _build_overrides(args)
    options = {
        "experiment_dir": args.experiment_dir,
        "checkpoint_type": args.checkpoint_type,
        "num_arch_samples": args.num_arch_samples,
        "sample_seed": args.sample_seed,
        "output_dir": args.output_dir,
        "dataset_root": args.dataset_root,
        "sintel_list": args.sintel_list,
        "sintel_patch_size": args.sintel_patch_size,
        "max_sintel_samples": args.max_sintel_samples,
        "bn_recal_batches": args.bn_recal_batches,
        "bn_recal_batch_size": args.bn_recal_batch_size,
        "eval_batches_per_arch": args.eval_batches_per_arch,
        "eval_batch_size": args.eval_batch_size,
        "gpu_device": args.gpu_device,
        "dry_run": args.dry_run,
    }
    result = run_rank_consistency_diagnostic(config_path=args.config, overrides=overrides, options=options)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
