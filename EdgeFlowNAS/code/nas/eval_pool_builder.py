"""验证子网池构建器。"""  # 定义模块用途

import random  # 导入随机模块
from typing import List  # 导入类型注解


def build_eval_pool(seed: int, size: int) -> List[List[int]]:  # 定义验证池构建函数
    """构建固定大小的验证子网池。"""  # 说明函数用途
    rng = random.Random(seed)  # 创建可复现随机数生成器
    pool: List[List[int]] = []  # 初始化子网池列表
    for _ in range(size):  # 按目标大小循环生成
        code = [rng.randint(0, 2) for _ in range(9)]  # 生成9维随机编码
        pool.append(code)  # 将编码加入子网池
    return pool  # 返回构建完成的子网池

