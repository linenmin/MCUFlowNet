"""CLI wrapper for retrain_v2 FT3D stage."""

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
    parser = argparse.ArgumentParser(description="Retrain V2 fixed candidates on FT3D")
    parser.add_argument("--config", default="configs/retrain_v2_ft3d.yaml", help="YAML config path")
    parser.add_argument("--arch_codes", default=None, help="11D arch codes joined by '+'")
    parser.add_argument("--model_names", default=None, help="Model names joined by '+'")
    parser.add_argument("--experiment_name", default=None, help="Experiment name")
    parser.add_argument("--fc2_experiment_dir", default=None, help="FC2 stage experiment directory for warm-start")
    parser.add_argument("--init_experiment_dir", default=None, help="Experiment directory used to initialize model checkpoints")
    parser.add_argument("--init_ckpt_name", default=None, help="Checkpoint name inside each model checkpoint directory")
    parser.add_argument("--base_path", default=None, help="Dataset base path")
    parser.add_argument("--frames_base_path", default=None, help="FT3D frames root")
    parser.add_argument("--flow_base_path", default=None, help="FT3D flow root")
    parser.add_argument("--ft3d_num_workers", type=int, default=None, help="FT3D batch loader worker count")
    parser.add_argument("--train_dir", default=None, help="FT3D train split path")
    parser.add_argument("--val_dir", default=None, help="FT3D val split path")
    parser.add_argument("--gpu_device", type=int, default=None, help="GPU index, -1=CPU")
    parser.add_argument("--num_epochs", type=int, default=None, help="Epoch count")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Initial LR")
    parser.add_argument("--lr_min", type=float, default=None, help="Minimum cosine LR")
    parser.add_argument("--resume", action="store_true", help="Resume current experiment")
    parser.add_argument("--resume_experiment_name", default=None, help="Resume experiment name")
    parser.add_argument("--resume_ckpt_name", default=None, help="Checkpoint name to restore when resuming")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config_path = args.config if os.path.isabs(args.config) else os.path.join(_PROJECT_ROOT, args.config)
    config = _load_yaml(config_path)
    if args.arch_codes is not None:
        config["arch_codes"] = args.arch_codes
    if args.model_names is not None:
        config["model_names"] = args.model_names
    if args.experiment_name is not None:
        config.setdefault("runtime", {})["experiment_name"] = args.experiment_name
    if args.seed is not None:
        config.setdefault("runtime", {})["seed"] = args.seed
    if args.fc2_experiment_dir is not None:
        config.setdefault("checkpoint", {})["init_mode"] = "experiment_dir"
        config.setdefault("checkpoint", {})["init_experiment_dir"] = args.fc2_experiment_dir
    if args.init_experiment_dir is not None:
        config.setdefault("checkpoint", {})["init_mode"] = "experiment_dir"
        config.setdefault("checkpoint", {})["init_experiment_dir"] = args.init_experiment_dir
    if args.init_ckpt_name is not None:
        config.setdefault("checkpoint", {})["init_ckpt_name"] = args.init_ckpt_name
    if args.base_path is not None:
        config.setdefault("data", {})["base_path"] = args.base_path
    if args.frames_base_path is not None:
        config.setdefault("data", {})["ft3d_frames_base_path"] = args.frames_base_path
        config.setdefault("data", {})["ft3d_frames_base_paths"] = [args.frames_base_path]
    if args.flow_base_path is not None:
        config.setdefault("data", {})["ft3d_flow_base_path"] = args.flow_base_path
    if args.ft3d_num_workers is not None:
        config.setdefault("data", {})["ft3d_num_workers"] = args.ft3d_num_workers
    if args.train_dir is not None:
        config.setdefault("data", {})["train_dir"] = args.train_dir
    if args.val_dir is not None:
        config.setdefault("data", {})["val_dir"] = args.val_dir
    if args.gpu_device is not None:
        config.setdefault("train", {})["gpu_device"] = args.gpu_device
    if args.num_epochs is not None:
        config.setdefault("train", {})["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        config.setdefault("train", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        config.setdefault("train", {})["lr"] = args.lr
    if args.lr_min is not None:
        config.setdefault("train", {})["lr_min"] = args.lr_min
    if args.resume:
        config.setdefault("checkpoint", {})["load_checkpoint"] = True
    if args.resume_experiment_name is not None:
        config.setdefault("checkpoint", {})["resume_experiment_name"] = args.resume_experiment_name
    if args.resume_ckpt_name is not None:
        config.setdefault("checkpoint", {})["resume_ckpt_name"] = args.resume_ckpt_name

    from efnas.engine.retrain_v2_trainer import train_retrain_v2

    return train_retrain_v2(config)


if __name__ == "__main__":
    raise SystemExit(main())
