"""梯度裁剪占位实现。"""  # 定义模块用途


def clip_global_norm(value: float, max_norm: float) -> float:  # 定义全局裁剪函数
    """裁剪梯度范数到指定上限。"""  # 说明函数用途
    return min(float(value), float(max_norm))  # 返回裁剪后的范数

