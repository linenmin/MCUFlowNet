"""180x240 变换工具。"""  # 定义模块用途

import numpy as np  # 导入NumPy模块


def standardize_image_tensor(image_tensor: np.ndarray) -> np.ndarray:  # 定义图像标准化函数
    """将图像张量从[0,255]映射到[-1,1]。"""  # 说明函数用途
    normalized = image_tensor.astype(np.float32) / 255.0  # 归一化到[0,1]
    centered = normalized - 0.5  # 平移到[-0.5,0.5]
    scaled = centered * 2.0  # 缩放到[-1,1]
    return scaled.astype(np.float32)  # 返回标准化张量

