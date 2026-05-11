"""CLI wrapper for one deploy-resolution fine-tune candidate."""

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
    parser = argparse.ArgumentParser(
        description="Fine-tune one deploy-resolution candidate (v3 subnet or mainline)."
    )
    parser.add_argument("--config", default="configs/retrain_v3_deploy_ft.yaml")
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--arch_family", required=True, choices=["fixed_v3", "edgeflownet_mainline"])
    parser.add_argument("--arch_code", default="")
    parser.add_argument("--init_mode", choices=["experiment_dir", "explicit_path"], default=None)
    parser.add_argument("--init_experiment_dir", default=None)
    parser.add_argument("--init_ckpt_name", default=None)
    parser.add_argument("--init_ckpt_path", default=None)
    parser.add_argument("--flow_divisor", type=float, default=None)
    parser.add_argument("--input_height", type=int, default=None)
    parser.add_argument("--input_width", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_schedule", default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--grad_clip_global_norm", type=float, default=None)
    parser.add_argument("--gpu_device", type=int, default=None)
    parser.add_argument("--base_path", default=None)
    parser.add_argument("--train_dir", default=None)
    parser.add_argument("--val_dir", default=None)
    parser.add_argument("--ft3d_num_workers", type=int, default=None)
    parser.add_argument("--ft3d_eval_num_workers", type=int, default=None)
    parser.add_argument("--prefetch_batches", type=int, default=None)
    parser.add_argument("--eval_prefetch_batches", type=int, default=None)
    parser.add_argument("--ft3d_eval_every_epoch", type=int, default=None)
    parser.add_argument("--sintel_eval_every_epoch", type=int, default=None)
    parser.add_argument("--sintel_max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def _set(config: dict, section: str, key: str, value) -> None:
    if value is None:
        return
    config.setdefault(section, {})[key] = value


def _apply_overrides(config: dict, args) -> dict:
    if args.experiment_name is not None:
        config.setdefault("runtime", {})["experiment_name"] = args.experiment_name
    if args.seed is not None:
        config.setdefault("runtime", {})["seed"] = args.seed

    config["model_name"] = args.model_name
    config["arch_family"] = args.arch_family
    if args.arch_code:
        config["arch_code"] = args.arch_code

    # checkpoint section
    if args.init_mode is not None:
        config.setdefault("checkpoint", {})["init_mode"] = args.init_mode
    _set(config, "checkpoint", "init_experiment_dir", args.init_experiment_dir)
    _set(config, "checkpoint", "init_ckpt_name", args.init_ckpt_name)
    _set(config, "checkpoint", "init_ckpt_path", args.init_ckpt_path)

    # data section
    _set(config, "data", "ft3d_flow_divisor", args.flow_divisor)
    _set(config, "data", "input_height", args.input_height)
    _set(config, "data", "input_width", args.input_width)
    _set(config, "data", "base_path", args.base_path)
    _set(config, "data", "train_dir", args.train_dir)
    _set(config, "data", "val_dir", args.val_dir)
    _set(config, "data", "ft3d_num_workers", args.ft3d_num_workers)
    _set(config, "data", "ft3d_eval_num_workers", args.ft3d_eval_num_workers)
    _set(config, "data", "prefetch_batches", args.prefetch_batches)
    _set(config, "data", "eval_prefetch_batches", args.eval_prefetch_batches)

    # train section
    _set(config, "train", "gpu_device", args.gpu_device)
    _set(config, "train", "num_epochs", args.num_epochs)
    _set(config, "train", "batch_size", args.batch_size)
    _set(config, "train", "micro_batch_size", args.micro_batch_size)
    _set(config, "train", "lr", args.lr)
    _set(config, "train", "lr_schedule", args.lr_schedule)
    _set(config, "train", "early_stop_patience", args.early_stop_patience)
    _set(config, "train", "grad_clip_global_norm", args.grad_clip_global_norm)

    # eval section
    _set(config, "eval", "eval_every_epoch", args.ft3d_eval_every_epoch)
    if args.sintel_eval_every_epoch is not None:
        config.setdefault("eval", {}).setdefault("sintel", {})["eval_every_epoch"] = args.sintel_eval_every_epoch
    if args.sintel_max_samples is not None:
        config.setdefault("eval", {}).setdefault("sintel", {})["max_samples"] = args.sintel_max_samples
    return config


def main() -> int:
    args = _build_parser().parse_args()
    config_path = (
        args.config if os.path.isabs(args.config) else os.path.join(_PROJECT_ROOT, args.config)
    )
    config = _apply_overrides(_load_yaml(config_path), args)
    from efnas.engine.deploy_ft_trainer import train_deploy_ft

    return train_deploy_ft(config)


if __name__ == "__main__":
    raise SystemExit(main())
