"""超网评估占位脚本。"""  # 定义模块用途

import argparse  # 导入参数解析模块
import json  # 导入JSON模块
from pathlib import Path  # 导入路径工具

import yaml  # 导入YAML模块

from code.nas.eval_pool_builder import build_eval_pool  # 导入验证池构建函数
from code.utils.json_io import write_json  # 导入JSON写入工具


def _load_config(path_like: str) -> dict:  # 定义配置加载函数
    """读取YAML配置文件。"""  # 说明函数用途
    with Path(path_like).open("r", encoding="utf-8") as handle:  # 以UTF-8打开文件
        return yaml.safe_load(handle)  # 解析并返回配置字典


def _build_parser() -> argparse.ArgumentParser:  # 定义参数解析器构建函数
    """创建命令行参数解析器。"""  # 说明函数用途
    parser = argparse.ArgumentParser(description="supernet eval placeholder")  # 创建解析器
    parser.add_argument("--config", required=True, help="path to yaml config")  # 添加配置参数
    parser.add_argument("--eval_only", action="store_true", help="enable eval-only flow")  # 添加仅评估开关
    parser.add_argument("--bn_recal_batches", type=int, default=8, help="bn recalibration batches")  # 添加BN重估参数
    return parser  # 返回解析器


def _build_bn_info(batches: int) -> dict:  # 定义BN摘要构建函数
    """构建NAS层可用的BN重估摘要。"""  # 说明函数用途
    return {  # 返回BN摘要字典
        "bn_recal_batches": int(batches),  # 写入BN重估批次数
        "bn_mean_shift": 0.0,  # 写入占位均值变化
        "bn_var_shift": 0.0,  # 写入占位方差变化
    }


def main() -> int:  # 定义主函数
    """执行评估占位流程。"""  # 说明函数用途
    parser = _build_parser()  # 构建参数解析器
    args = parser.parse_args()  # 解析命令行参数
    config = _load_config(args.config)  # 加载配置字典
    if not args.eval_only:  # 判断是否启用eval_only
        parser.error("placeholder evaluator requires --eval_only")  # 提示必须开启eval_only
    pool_size = int(config.get("eval", {}).get("eval_pool_size", 12))  # 读取验证池大小
    seed = int(config.get("runtime", {}).get("seed", 42))  # 读取随机种子
    pool = build_eval_pool(seed=seed, size=pool_size)  # 构建固定验证子网池
    bn_info = _build_bn_info(batches=args.bn_recal_batches)  # 构建占位BN重估摘要
    output_root = Path(config.get("runtime", {}).get("output_root", "outputs/supernet"))  # 读取输出目录
    output_root.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
    eval_pool_path = output_root / "eval_pool_12.json"  # 计算验证池输出路径
    write_json(str(eval_pool_path), {"pool": pool, "bn_recal": bn_info})  # 写入验证池与BN信息
    print(json.dumps({"status": "ok", "eval_pool_path": str(eval_pool_path)}, ensure_ascii=False, indent=2))  # 打印执行摘要
    return 0  # 返回成功状态


if __name__ == "__main__":  # 判断是否为脚本直运行
    raise SystemExit(main())  # 以主函数返回码退出
