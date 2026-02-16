"""180x240 变换占位定义。"""  # 定义模块用途

from typing import Dict  # 导入类型注解


def build_transforms(height: int, width: int) -> Dict[str, int]:  # 定义变换构建函数
    """构建固定分辨率变换描述。"""  # 说明函数用途
    return {"height": int(height), "width": int(width)}  # 返回占位变换配置

