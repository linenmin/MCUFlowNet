"""Supernet 训练应用编排层。"""  # 定义模块用途

from copy import deepcopy  # 导入深拷贝工具
from pathlib import Path  # 导入路径工具
from typing import Any, Dict  # 导入类型注解

import yaml  # 导入YAML解析模块

from code.engine.supernet_trainer import train_supernet  # 导入训练执行函数
from code.utils.path_utils import project_root  # 导入项目根目录函数


def _load_yaml(config_path: str) -> Dict[str, Any]:  # 定义YAML加载函数
    """读取YAML配置文件。"""  # 说明函数用途
    path = Path(config_path)  # 构造配置路径对象
    if not path.is_absolute():  # 判断配置路径是否为绝对路径
        path = project_root() / path  # 将相对路径转换为项目内绝对路径
    with path.open("r", encoding="utf-8") as handle:  # 以UTF-8打开配置文件
        content = yaml.safe_load(handle)  # 安全解析YAML内容
    if not isinstance(content, dict):  # 检查解析结果是否为字典
        raise ValueError("配置文件顶层必须是字典结构。")  # 抛出结构错误
    return content  # 返回配置字典


def _set_nested(config: Dict[str, Any], key_path: str, value: Any) -> None:  # 定义嵌套写入函数
    """按点号路径写入嵌套配置项。"""  # 说明函数用途
    keys = key_path.split(".")  # 拆分层级路径
    cursor = config  # 初始化游标对象
    for key in keys[:-1]:  # 遍历除最后一层外的路径键
        if key not in cursor or not isinstance(cursor[key], dict):  # 判断下一层字典是否存在
            cursor[key] = {}  # 不存在时创建字典层
        cursor = cursor[key]  # 下钻到下一层
    cursor[keys[-1]] = value  # 写入最终键的值


def _merge_overrides(base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:  # 定义覆写合并函数
    """合并命令行覆写到基础配置。"""  # 说明函数用途
    merged = deepcopy(base_config)  # 深拷贝基础配置避免原地修改
    for key_path, value in overrides.items():  # 遍历每个覆写项
        _set_nested(merged, key_path, value)  # 写入单个覆写值
    return merged  # 返回合并后的配置


def run_supernet_app(config_path: str, overrides: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:  # 定义应用入口函数
    """执行Supernet训练应用流程。"""  # 说明函数用途
    base_config = _load_yaml(config_path)  # 读取基础配置
    final_config = _merge_overrides(base_config, overrides)  # 合并命令行覆写
    if dry_run:  # 判断是否为干跑模式
        return {"exit_code": 0, "config": final_config}  # 返回配置预览结果
    exit_code = train_supernet(final_config)  # 调用训练执行层
    return {"exit_code": int(exit_code), "config": final_config}  # 返回执行结果

