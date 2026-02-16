"""Supernet 训练命令入口。"""  # 定义脚本用途

import argparse  # 导入参数解析模块
import json  # 导入JSON模块
import sys  # 导入系统模块
from pathlib import Path  # 导入路径工具
from typing import Any, Dict  # 导入类型注解

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 计算项目根目录
if str(PROJECT_ROOT) not in sys.path:  # 检查项目根目录是否已在导入路径中
    sys.path.insert(0, str(PROJECT_ROOT))  # 将项目根目录加入导入路径

from code.app.train_supernet_app import run_supernet_app  # 导入应用层训练入口


def _put_override(overrides: Dict[str, Any], key: str, value: Any) -> None:  # 定义覆写写入函数
    """按键路径写入覆写参数。"""  # 说明函数用途
    if value is None:  # 判断当前值是否为空
        return  # 空值时不写入覆写
    overrides[key] = value  # 写入覆写字典


def _build_overrides(args: argparse.Namespace) -> Dict[str, Any]:  # 定义参数覆写构建函数
    """从命令行参数构建配置覆写。"""  # 说明函数用途
    overrides: Dict[str, Any] = {}  # 初始化覆写字典
    _put_override(overrides, "train.gpu_device", args.gpu_device)  # 写入GPU覆写
    _put_override(overrides, "train.num_epochs", args.num_epochs)  # 写入轮数覆写
    _put_override(overrides, "train.steps_per_epoch", args.steps_per_epoch)  # 写入每轮步数覆写
    _put_override(overrides, "train.batch_size", args.batch_size)  # 写入批大小覆写
    _put_override(overrides, "train.micro_batch_size", args.micro_batch_size)  # 写入微批大小覆写
    _put_override(overrides, "train.lr", args.lr)  # 写入学习率覆写
    _put_override(overrides, "train.supernet_mode", args.supernet_mode)  # 写入超网模式覆写
    _put_override(overrides, "runtime.experiment_name", args.experiment_name)  # 写入实验名覆写
    _put_override(overrides, "data.dataset", args.dataset)  # 写入数据集覆写
    _put_override(overrides, "data.data_list", args.data_list)  # 写入数据列表覆写
    _put_override(overrides, "data.base_path", args.base_path)  # 写入数据根路径覆写
    _put_override(overrides, "checkpoint.resume_experiment_name", args.resume_experiment_name)  # 写入恢复实验名覆写
    if args.fast_mode:  # 判断是否开启快速模式
        _put_override(overrides, "train.fast_mode", True)  # 写入快速模式覆写
    if args.load_checkpoint:  # 判断是否加载断点
        _put_override(overrides, "checkpoint.load_checkpoint", True)  # 写入加载断点覆写
    return overrides  # 返回覆写字典


def _build_parser() -> argparse.ArgumentParser:  # 定义参数解析器构建函数
    """创建命令行参数解析器。"""  # 说明函数用途
    parser = argparse.ArgumentParser(description="run supernet train locally without docker")  # 创建解析器
    parser.add_argument("--config", default="configs/supernet_fc2_180x240.yaml", help="path to supernet config yaml")  # 添加配置路径参数
    parser.add_argument("--gpu_device", type=int, default=None, help="GPU index, set -1 for CPU")  # 添加GPU参数
    parser.add_argument("--num_epochs", type=int, default=None, help="number of training epochs")  # 添加轮数参数
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="steps per epoch")  # 添加每轮步数参数
    parser.add_argument("--batch_size", type=int, default=None, help="mini-batch size")  # 添加批大小参数
    parser.add_argument("--micro_batch_size", type=int, default=None, help="micro batch size for gradient accumulation")  # 添加微批大小参数
    parser.add_argument("--lr", type=float, default=None, help="learning rate")  # 添加学习率参数
    parser.add_argument("--dataset", default=None, help="dataset name, e.g. FC2")  # 添加数据集参数
    parser.add_argument("--data_list", default=None, help="directory containing list files")  # 添加数据列表参数
    parser.add_argument("--base_path", default=None, help="optional dataset base path")  # 添加数据根路径参数
    parser.add_argument("--experiment_name", default=None, help="experiment name under outputs")  # 添加实验名参数
    parser.add_argument("--resume_experiment_name", default=None, help="resume source experiment name")  # 添加恢复实验名参数
    parser.add_argument("--supernet_mode", default=None, help="supernet mode name")  # 添加超网模式参数
    parser.add_argument("--load_checkpoint", action="store_true", help="resume from checkpoint")  # 添加加载断点开关
    parser.add_argument("--fast_mode", action="store_true", help="enable fast mode")  # 添加快速模式开关
    parser.add_argument("--dry_run", action="store_true", help="print merged config and exit")  # 添加干跑开关
    return parser  # 返回解析器


def main() -> int:  # 定义主函数
    """执行命令入口逻辑。"""  # 说明函数用途
    parser = _build_parser()  # 构建参数解析器
    args = parser.parse_args()  # 解析命令行参数
    overrides = _build_overrides(args)  # 构建配置覆写
    result = run_supernet_app(config_path=args.config, overrides=overrides, dry_run=args.dry_run)  # 调用应用层入口
    if args.dry_run:  # 判断是否为干跑模式
        print(json.dumps(result, ensure_ascii=False, indent=2))  # 打印合并后的配置
        return 0  # 返回成功状态
    print(f"Supernet training finished with exit_code={result.get('exit_code', 1)}")  # 打印结束信息
    return int(result.get("exit_code", 1))  # 返回训练退出码


if __name__ == "__main__":  # 判断是否为脚本直运行
    raise SystemExit(main())  # 以主函数返回码退出
