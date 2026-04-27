"""CLI wrapper for Ablation V1 FT3D stage."""

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
    parser = argparse.ArgumentParser(description="Run Ablation V1 on FT3D")
    parser.add_argument("--config", default="configs/ablation_v1_ft3d.yaml")
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--init_experiment_dir", default=None)
    parser.add_argument("--init_ckpt_name", default=None)
    parser.add_argument("--gpu_device", type=int, default=None)
    parser.add_argument("--base_path", default=None)
    parser.add_argument("--frames_base_path", default=None)
    parser.add_argument("--flow_base_path", default=None)
    parser.add_argument("--train_dir", default=None)
    parser.add_argument("--val_dir", default=None)
    parser.add_argument("--ft3d_num_workers", type=int, default=None)
    parser.add_argument("--ft3d_eval_num_workers", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_min", type=float, default=None)
    parser.add_argument("--grad_clip_global_norm", type=float, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--early_stop_min_delta", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_experiment_name", default=None)
    parser.add_argument("--resume_ckpt_name", default=None)
    parser.add_argument("--allow_config_mismatch", action="store_true")
    parser.add_argument("--variants", default=None, help="Comma-separated ablation variant names to run")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def _filter_variants(config: dict, variant_names: str) -> None:
    requested = [item.strip() for item in str(variant_names).replace("+", ",").split(",") if item.strip()]
    if not requested:
        return
    variants = list(config.get("variants", []))
    by_name = {str(item.get("name")): item for item in variants}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"unknown ablation variant(s): {', '.join(missing)}")
    config["variants"] = [by_name[name] for name in requested]


def main() -> int:
    args = _build_parser().parse_args()
    config_path = args.config if os.path.isabs(args.config) else os.path.join(_PROJECT_ROOT, args.config)
    config = _load_yaml(config_path)
    if args.experiment_name is not None:
        config.setdefault("runtime", {})["experiment_name"] = args.experiment_name
    if args.seed is not None:
        config.setdefault("runtime", {})["seed"] = args.seed
    if args.init_experiment_dir is not None:
        config.setdefault("checkpoint", {})["init_mode"] = "experiment_dir"
        config.setdefault("checkpoint", {})["init_experiment_dir"] = args.init_experiment_dir
    if args.init_ckpt_name is not None:
        config.setdefault("checkpoint", {})["init_ckpt_name"] = args.init_ckpt_name
    if args.gpu_device is not None:
        config.setdefault("train", {})["gpu_device"] = args.gpu_device
    for key in ("num_epochs", "batch_size", "micro_batch_size", "lr", "lr_min", "grad_clip_global_norm", "early_stop_patience", "early_stop_min_delta"):
        value = getattr(args, key)
        if value is not None:
            config.setdefault("train", {})[key] = value
    if args.base_path is not None:
        config.setdefault("data", {})["base_path"] = args.base_path
    if args.frames_base_path is not None:
        config.setdefault("data", {})["ft3d_frames_base_paths"] = [args.frames_base_path]
    if args.flow_base_path is not None:
        config.setdefault("data", {})["ft3d_flow_base_path"] = args.flow_base_path
    if args.train_dir is not None:
        config.setdefault("data", {})["train_dir"] = args.train_dir
    if args.val_dir is not None:
        config.setdefault("data", {})["val_dir"] = args.val_dir
    if args.ft3d_num_workers is not None:
        config.setdefault("data", {})["ft3d_num_workers"] = args.ft3d_num_workers
    if args.ft3d_eval_num_workers is not None:
        config.setdefault("data", {})["ft3d_eval_num_workers"] = args.ft3d_eval_num_workers
    if args.resume:
        config.setdefault("checkpoint", {})["load_checkpoint"] = True
    if args.resume_experiment_name is not None:
        config.setdefault("checkpoint", {})["resume_experiment_name"] = args.resume_experiment_name
    if args.resume_ckpt_name is not None:
        config.setdefault("checkpoint", {})["resume_ckpt_name"] = args.resume_ckpt_name
    if args.allow_config_mismatch:
        config.setdefault("checkpoint", {})["allow_config_mismatch"] = True
    if args.variants is not None:
        _filter_variants(config, args.variants)

    from efnas.engine.ablation_v1_trainer import train_ablation_v1

    return train_ablation_v1(config)


if __name__ == "__main__":
    raise SystemExit(main())
