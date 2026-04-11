"""CLI wrapper for validating NSGA-II Pareto and near-Pareto subnets on Sintel."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from efnas.nas.supernet_v2_pareto_sintel_validation import run_pareto_sintel_validation


def _build_parser() -> argparse.ArgumentParser:
    """Build wrapper CLI parser."""
    parser = argparse.ArgumentParser(description="validate FC2 Pareto and near-Pareto V2 subnets on Sintel")
    parser.add_argument("--config", default="configs/supernet_fc2_172x224_v2.yaml")
    parser.add_argument("--history_csv", required=True)
    parser.add_argument("--experiment_dir", default=None)
    parser.add_argument("--checkpoint_type", choices=["best", "last"], default="best")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--sintel_list", default="EdgeFlowNet/code/dataset_paths/MPI_Sintel_Final_train_list.txt")
    parser.add_argument("--sintel_patch_size", default="416,1024")
    parser.add_argument("--max_sintel_samples", type=int, default=None)
    parser.add_argument("--bn_recal_batches", type=int, default=16)
    parser.add_argument("--bn_recal_batch_size", type=int, default=None)
    parser.add_argument("--gpu_device", type=int, default=None)
    parser.add_argument("--near_rel_gap", type=float, default=0.05)
    parser.add_argument("--max_near", type=int, default=20)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    result = run_pareto_sintel_validation(
        config_path=args.config,
        overrides={},
        options={
            "history_csv": args.history_csv,
            "experiment_dir": args.experiment_dir,
            "checkpoint_type": args.checkpoint_type,
            "output_dir": args.output_dir,
            "dataset_root": args.dataset_root,
            "sintel_list": args.sintel_list,
            "sintel_patch_size": args.sintel_patch_size,
            "max_sintel_samples": args.max_sintel_samples,
            "bn_recal_batches": args.bn_recal_batches,
            "bn_recal_batch_size": args.bn_recal_batch_size,
            "gpu_device": args.gpu_device,
            "near_rel_gap": args.near_rel_gap,
            "max_near": args.max_near,
            "dry_run": args.dry_run,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
