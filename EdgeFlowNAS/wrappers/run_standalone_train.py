"""单模型/多模型独立重训 CLI 入口。

用法示例:
  # 单模型训练
  python wrappers/run_standalone_train.py \\
    --config configs/standalone_fc2_180x240.yaml \\
    --arch_codes "0,2,1,1,0,0,1,0,1" \\
    --experiment_name retrain_run1

  # 双模型对比训练
  python wrappers/run_standalone_train.py \\
    --config configs/standalone_fc2_180x240.yaml \\
    --arch_codes "0,2,1,1,0,0,1,0,1+0,0,0,0,2,1,2,2,2" \\
    --arch_names "target+baseline" \\
    --experiment_name retrain_dual_run1
"""

import argparse
import os
import sys

# 将项目根目录加入 sys.path（EdgeFlowNAS/）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _load_yaml(path: str) -> dict:
    """加载 YAML 配置文件。"""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并两个字典，override 覆盖 base。"""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(
        description="单模型/多模型独立重训",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- 配置文件 ---
    parser.add_argument("--config", default="configs/standalone_fc2_180x240.yaml",
                        help="YAML 配置文件路径")

    # --- 必选: 架构码 ---
    parser.add_argument("--arch_codes", required=True,
                        help="架构码 (9位逗号分隔, 多个用+连接, "
                             "例如 '0,2,1,1,0,0,1,0,1' 或 '0,2,1,1,0,0,1,0,1+0,0,0,0,2,1,2,2,2')")
    parser.add_argument("--arch_names", default="",
                        help="架构名 (与 arch_codes 对应, 用+连接, "
                             "例如 'target' 或 'target+baseline')")

    # --- 运行时参数（CLI 覆盖 YAML） ---
    parser.add_argument("--experiment_name", default=None, help="实验名称")
    parser.add_argument("--base_path", default=None, help="数据集根路径")
    parser.add_argument("--train_dir", default=None, help="训练集目录")
    parser.add_argument("--val_dir", default=None, help="验证集目录")
    parser.add_argument("--gpu_device", type=int, default=None, help="GPU 编号, -1=CPU")
    parser.add_argument("--num_epochs", type=int, default=None, help="训练 epoch 数")
    parser.add_argument("--batch_size", type=int, default=None, help="batch size")
    parser.add_argument("--lr", type=float, default=None, help="初始学习率")
    parser.add_argument("--lr_min", type=float, default=None, help="cosine 最低学习率")
    parser.add_argument("--eval_every_epoch", type=int, default=None, help="每 N epoch 评估一次")
    parser.add_argument("--load_checkpoint", action="store_true", help="从 checkpoint 恢复训练")
    parser.add_argument("--resume_experiment_name", default=None, help="恢复源实验名")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")

    args = parser.parse_args()

    # --- 加载基础配置 ---
    config_path = os.path.join(_PROJECT_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    if os.path.exists(config_path):
        config = _load_yaml(config_path)
    else:
        print(f"WARNING: 配置文件 {config_path} 不存在，使用默认空配置")
        config = {}

    # --- CLI 参数覆盖 YAML ---
    # 顶级参数
    config["arch_codes"] = args.arch_codes
    if args.arch_names:
        config["arch_names"] = args.arch_names

    # runtime
    if args.experiment_name is not None:
        config.setdefault("runtime", {})["experiment_name"] = args.experiment_name
    if args.seed is not None:
        config.setdefault("runtime", {})["seed"] = args.seed

    # train
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

    # data
    if args.base_path is not None:
        config.setdefault("data", {})["base_path"] = args.base_path
    if args.train_dir is not None:
        config.setdefault("data", {})["train_dir"] = args.train_dir
    if args.val_dir is not None:
        config.setdefault("data", {})["val_dir"] = args.val_dir

    # eval
    if args.eval_every_epoch is not None:
        config.setdefault("eval", {})["eval_every_epoch"] = args.eval_every_epoch

    # checkpoint
    if args.load_checkpoint:
        config.setdefault("checkpoint", {})["load_checkpoint"] = True
    if args.resume_experiment_name is not None:
        config.setdefault("checkpoint", {})["resume_experiment_name"] = args.resume_experiment_name

    # --- 执行训练 ---
    from code.engine.standalone_trainer import train_standalone
    exit_code = train_standalone(config)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
