"""Checkpoint 管理工具。"""  # 定义模块用途

from pathlib import Path  # 导入路径工具
from typing import Dict  # 导入类型注解

from code.utils.json_io import write_json  # 导入JSON写入工具


def build_checkpoint_paths(experiment_dir: str) -> Dict[str, Path]:  # 定义checkpoint路径构建函数
    """构建best与last的占位路径。"""  # 说明函数用途
    root = Path(experiment_dir) / "checkpoints"  # 计算checkpoint目录路径
    root.mkdir(parents=True, exist_ok=True)  # 确保checkpoint目录存在
    return {  # 返回路径字典
        "best": root / "supernet_best.ckpt",  # 返回best路径
        "last": root / "supernet_last.ckpt",  # 返回last路径
    }


def write_checkpoint_placeholder(path: Path, payload: Dict) -> None:  # 定义checkpoint占位写入函数
    """写入轻量占位checkpoint文件。"""  # 说明函数用途
    write_json(str(path), payload)  # 写入JSON占位内容

