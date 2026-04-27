"""CLI wrapper for Ablation V1 FC2 stage."""

import argparse
import os
import sys


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _load_yaml(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Ablation V1 on FC2")
    parser.add_argument("--config", default="configs/ablation_v1_fc2.yaml")
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--gpu_device", type=int, default=None)
    parser.add_argument("--base_path", default=None)
    parser.add_argument("--train_dir", default=None)
    parser.add_argument("--val_dir", default=None)
    parser.add_argument("--fc2_num_workers", type=int, default=None)
    parser.add_argument("--fc2_eval_num_workers", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_min", type=float, default=None)
    parser.add_argument("--grad_clip_global_norm", type=float, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--early_stop_min_delta", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_experiment_name", default=None)
    parser.add_argument("--resume_ckpt_name", default=None)
    parser.add_argument("--allow_config_mismatch", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def _apply_common_overrides(config: dict, args) -> dict:
    if args.experiment_name is not None:
        config.setdefault("runtime", {})["experiment_name"] = args.experiment_name
    if args.seed is not None:
        config.setdefault("runtime", {})["seed"] = args.seed
    if args.gpu_device is not None:
        config.setdefault("train", {})["gpu_device"] = args.gpu_device
    for key in ("num_epochs", "batch_size", "lr", "lr_min", "grad_clip_global_norm", "early_stop_patience", "early_stop_min_delta"):
        value = getattr(args, key)
        if value is not None:
            config.setdefault("train", {})[key] = value
    if args.base_path is not None:
        config.setdefault("data", {})["base_path"] = args.base_path
    if args.train_dir is not None:
        config.setdefault("data", {})["train_dir"] = args.train_dir
    if args.val_dir is not None:
        config.setdefault("data", {})["val_dir"] = args.val_dir
    if args.fc2_num_workers is not None:
        config.setdefault("data", {})["fc2_num_workers"] = args.fc2_num_workers
    if args.fc2_eval_num_workers is not None:
        config.setdefault("data", {})["fc2_eval_num_workers"] = args.fc2_eval_num_workers
    if args.resume:
        config.setdefault("checkpoint", {})["load_checkpoint"] = True
    if args.resume_experiment_name is not None:
        config.setdefault("checkpoint", {})["resume_experiment_name"] = args.resume_experiment_name
    if args.resume_ckpt_name is not None:
        config.setdefault("checkpoint", {})["resume_ckpt_name"] = args.resume_ckpt_name
    if args.allow_config_mismatch:
        config.setdefault("checkpoint", {})["allow_config_mismatch"] = True
    return config


def main() -> int:
    args = _build_parser().parse_args()
    config_path = args.config if os.path.isabs(args.config) else os.path.join(_PROJECT_ROOT, args.config)
    config = _apply_common_overrides(_load_yaml(config_path), args)
    from efnas.engine.ablation_v1_trainer import train_ablation_v1

    return train_ablation_v1(config)


if __name__ == "__main__":
    raise SystemExit(main())
