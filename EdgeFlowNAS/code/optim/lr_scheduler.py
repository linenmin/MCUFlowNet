"""学习率调度占位实现。"""  # 定义模块用途

import math  # 导入数学模块


def cosine_lr(base_lr: float, step_idx: int, total_steps: int) -> float:  # 定义余弦学习率函数
    """计算余弦退火学习率。"""  # 说明函数用途
    if total_steps <= 0:  # 判断总步数是否合法
        return float(base_lr)  # 非法时回退基础学习率
    ratio = min(max(step_idx / float(total_steps), 0.0), 1.0)  # 计算归一化进度
    return float(base_lr) * 0.5 * (1.0 + math.cos(math.pi * ratio))  # 返回余弦学习率

