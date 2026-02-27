"""头部块占位定义。"""  # 定义模块用途

from typing import Dict  # 导入类型注解


def build_head_block(kernel_label: str) -> Dict[str, str]:  # 定义头部块构建函数
    """根据卷积核标签构建头部块描述。"""  # 说明函数用途
    return {"kernel": kernel_label}  # 返回占位头部块字典

