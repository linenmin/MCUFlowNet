"""Supernet 训练应用编排层。"""  # 定义模块用途

from copy import deepcopy  # 导入深拷贝工具
from pathlib import Path  # 导入路径工具
from typing import Any, Dict  # 导入类型注解

try:  # 尝试导入YAML解析模块
    import yaml  # 导入YAML解析模块
except Exception:  # 捕获YAML导入异常
    yaml = None  # 回退为空解析器占位

from code.utils.path_utils import project_root  # 导入项目根目录函数


def _parse_scalar(value_text: str) -> Any:  # 定义标量解析函数
    """将YAML标量文本解析为Python值。"""  # 说明函数用途
    lowered = value_text.lower()  # 生成小写文本副本
    if lowered == "true":  # 判断是否为布尔真
        return True  # 返回布尔真
    if lowered == "false":  # 判断是否为布尔假
        return False  # 返回布尔假
    if value_text.startswith('"') and value_text.endswith('"'):  # 判断是否为双引号字符串
        return value_text[1:-1]  # 返回去引号字符串
    if value_text.startswith("'") and value_text.endswith("'"):  # 判断是否为单引号字符串
        return value_text[1:-1]  # 返回去引号字符串
    try:  # 尝试解析整数
        return int(value_text)  # 返回整数值
    except Exception:  # 捕获整数解析异常
        pass  # 跳过到下一个解析分支
    try:  # 尝试解析浮点数
        return float(value_text)  # 返回浮点值
    except Exception:  # 捕获浮点解析异常
        pass  # 跳过到字符串分支
    return value_text  # 返回原始字符串值


def _load_simple_yaml(path: Path) -> Dict[str, Any]:  # 定义轻量YAML解析函数
    """在无PyYAML环境下解析简化YAML配置。"""  # 说明函数用途
    root: Dict[str, Any] = {}  # 初始化根字典
    stack = [(-1, root)]  # 初始化缩进栈
    with path.open("r", encoding="utf-8") as handle:  # 以UTF-8打开配置文件
        for raw in handle:  # 遍历配置文件每行
            content = raw.split("#", 1)[0].rstrip("\n")  # 去除注释与换行
            if not content.strip():  # 判断是否为空内容行
                continue  # 空内容时跳过当前行
            indent = len(content) - len(content.lstrip(" "))  # 计算当前行缩进
            stripped = content.strip()  # 提取去空白后的内容
            while stack and indent <= stack[-1][0]:  # 弹出不匹配缩进层
                stack.pop()  # 弹出当前栈顶层
            parent = stack[-1][1]  # 获取当前父级字典
            if stripped.endswith(":"):  # 判断是否为字典起始行
                key = stripped[:-1].strip()  # 提取字典键名
                node: Dict[str, Any] = {}  # 初始化子字典
                parent[key] = node  # 将子字典挂载到父级
                stack.append((indent, node))  # 将子层压入缩进栈
                continue  # 进入下一行解析
            key, value_text = stripped.split(":", 1)  # 拆分键和值文本
            parent[key.strip()] = _parse_scalar(value_text.strip())  # 解析并写入标量值
    return root  # 返回解析后的配置字典


def _load_yaml(config_path: str) -> Dict[str, Any]:  # 定义YAML加载函数
    """读取YAML配置文件。"""  # 说明函数用途
    path = Path(config_path)  # 构造配置路径对象
    if not path.is_absolute():  # 判断配置路径是否为绝对路径
        path = project_root() / path  # 将相对路径转换为项目内绝对路径
    if yaml is None:  # 判断是否缺失PyYAML依赖
        content = _load_simple_yaml(path)  # 使用轻量解析器读取配置
    else:  # 使用标准PyYAML解析器
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
    from code.engine.supernet_trainer import train_supernet  # 延迟导入训练执行函数
    exit_code = train_supernet(final_config)  # 调用训练执行层
    return {"exit_code": int(exit_code), "config": final_config}  # 返回执行结果
