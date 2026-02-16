"""随机种子工具。"""  # 定义模块用途

import os  # 导入系统环境模块
import random  # 导入随机模块


def set_global_seed(seed: int) -> None:  # 定义全局种子设置函数
    """设置Python与环境级随机种子。"""  # 说明函数用途
    random.seed(seed)  # 设置Python随机种子
    os.environ["PYTHONHASHSEED"] = str(seed)  # 设置哈希随机种子
    try:  # 尝试导入NumPy
        import numpy as np  # 延迟导入NumPy
    except Exception:  # 捕获NumPy缺失异常
        return  # 无NumPy时直接返回
    np.random.seed(seed)  # 设置NumPy随机种子

