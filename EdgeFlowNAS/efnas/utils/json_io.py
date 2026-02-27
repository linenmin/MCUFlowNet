"""JSON读写工具。"""  # 定义模块用途

import json  # 导入JSON模块
from pathlib import Path  # 导入路径工具
from typing import Any  # 导入类型注解


def read_json(path_like: str) -> Any:  # 定义JSON读取函数
    """读取JSON文件并返回对象。"""  # 说明函数用途
    path = Path(path_like)  # 构造路径对象
    with path.open("r", encoding="utf-8") as handle:  # 以UTF-8打开文件
        return json.load(handle)  # 解析并返回JSON对象


def write_json(path_like: str, payload: Any) -> None:  # 定义JSON写入函数
    """将对象写入JSON文件。"""  # 说明函数用途
    path = Path(path_like)  # 构造路径对象
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保父目录存在
    with path.open("w", encoding="utf-8") as handle:  # 以UTF-8打开文件
        json.dump(payload, handle, ensure_ascii=False, indent=2)  # 写入格式化JSON

