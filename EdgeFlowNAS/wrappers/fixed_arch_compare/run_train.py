"""CLI entry for fixed-arch joint comparison training."""

import argparse
import os
import sys


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _load_yaml(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main():
    parser = argparse.ArgumentParser(
        description="Fixed-arch joint comparison training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/fixed_arch_compare_fc2_172x224.yaml",
        help="YAML config path",
    )
    parser.add_argument(
        "--backbone_arch_code",
        default=None,
        help="Fixed backbone arch code, e.g. '0,2,1,1,0,0,0,0,0'",
    )
    parser.add_argument(
        "--model_variants",
        default=None,
        help="Variants joined by '+', e.g. 'baseline+globalgate4x_bneckeca+globalgate4x_bneckeca_skip8x4x2x'",
    )
    parser.add_argument(
        "--model_names",
        default=None,
        help="Model names joined by '+', e.g. 'baseline+ablation+full'",
    )
    parser.add_argument("--experiment_name", default=None, help="Experiment name")
    parser.add_argument("--base_path", default=None, help="Dataset base path")
    parser.add_argument("--train_dir", default=None, help="Train split path")
    parser.add_argument("--val_dir", default=None, help="Val split path")
    parser.add_argument("--gpu_device", type=int, default=None, help="GPU index, -1=CPU")
    parser.add_argument("--num_epochs", type=int, default=None, help="Epoch count")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Initial LR")
    parser.add_argument("--lr_min", type=float, default=None, help="Minimum cosine LR")
    parser.add_argument("--eval_every_epoch", type=int, default=None, help="Evaluate every N epochs")
    parser.add_argument("--load_checkpoint", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--resume_experiment_name", default=None, help="Resume experiment name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    config_path = args.config if os.path.isabs(args.config) else os.path.join(_PROJECT_ROOT, args.config)
    config = _load_yaml(config_path)

    if args.backbone_arch_code is not None:
        config["backbone_arch_code"] = args.backbone_arch_code
    if args.model_variants is not None:
        config["model_variants"] = args.model_variants
    if args.model_names is not None:
        config["model_names"] = args.model_names

    if args.experiment_name is not None:
        config.setdefault("runtime", {})["experiment_name"] = args.experiment_name
    if args.seed is not None:
        config.setdefault("runtime", {})["seed"] = args.seed

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

    if args.base_path is not None:
        config.setdefault("data", {})["base_path"] = args.base_path
    if args.train_dir is not None:
        config.setdefault("data", {})["train_dir"] = args.train_dir
    if args.val_dir is not None:
        config.setdefault("data", {})["val_dir"] = args.val_dir

    if args.eval_every_epoch is not None:
        config.setdefault("eval", {})["eval_every_epoch"] = args.eval_every_epoch

    if args.load_checkpoint:
        config.setdefault("checkpoint", {})["load_checkpoint"] = True
    if args.resume_experiment_name is not None:
        config.setdefault("checkpoint", {})["resume_experiment_name"] = args.resume_experiment_name

    from efnas.engine.fixed_arch_compare_trainer import train_fixed_arch_compare

    sys.exit(train_fixed_arch_compare(config))


if __name__ == "__main__":
    main()
