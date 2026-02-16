"""Bilinear Supernet 网络占位定义。"""  # 定义模块用途

from typing import Dict, List  # 导入类型注解

from code.nas.arch_codec import decode_arch_code  # 导入架构解码函数


class MultiScaleResNetSupernet:  # 定义超网占位类
    """用于承载 arch_code 的占位超网结构。"""  # 说明类用途

    def __init__(self) -> None:  # 定义初始化函数
        self.name = "MultiScaleResNetSupernet"  # 记录模型名称

    def forward(self, arch_code: List[int]) -> Dict[str, Dict]:  # 定义前向占位函数
        """根据架构编码返回占位输出结构。"""  # 说明函数用途
        decoded = decode_arch_code(arch_code)  # 解码架构编码
        return {  # 返回占位输出
            "decoded_arch": decoded,  # 返回解码结构
            "output_shapes": {  # 返回多尺度输出形状
                "out_1_4": [45, 60],  # 返回1/4尺度尺寸
                "out_1_2": [90, 120],  # 返回1/2尺度尺寸
                "out_1_1": [180, 240],  # 返回全尺度尺寸
            },
        }

