"""Supernet 输出形状检查脚本。"""  # 定义模块用途

import argparse  # 导入参数解析模块
import json  # 导入JSON模块
import random  # 导入随机模块

from code.network.MultiScaleResNet_supernet import MultiScaleResNetSupernet  # 导入超网占位类


def _build_parser() -> argparse.ArgumentParser:  # 定义参数解析器构建函数
    """创建命令行参数解析器。"""  # 说明函数用途
    parser = argparse.ArgumentParser(description="model shape check placeholder")  # 创建解析器
    parser.add_argument("--h", type=int, default=180, help="input height")  # 添加高度参数
    parser.add_argument("--w", type=int, default=240, help="input width")  # 添加宽度参数
    parser.add_argument("--batch", type=int, default=2, help="batch size")  # 添加批大小参数
    parser.add_argument("--samples", type=int, default=20, help="number of random samples")  # 添加采样数参数
    parser.add_argument("--seed", type=int, default=42, help="random seed")  # 添加随机种子参数
    return parser  # 返回解析器


def main() -> int:  # 定义主函数
    """执行形状检查占位流程。"""  # 说明函数用途
    parser = _build_parser()  # 构建参数解析器
    args = parser.parse_args()  # 解析命令行参数
    rng = random.Random(args.seed)  # 创建可复现随机数生成器
    model = MultiScaleResNetSupernet()  # 创建超网占位模型
    last_shapes = None  # 初始化最后一次输出形状
    for _ in range(args.samples):  # 按采样数循环
        arch_code = [rng.randint(0, 2) for _ in range(9)]  # 随机生成9维编码
        output = model.forward(arch_code=arch_code)  # 执行占位前向
        last_shapes = output["output_shapes"]  # 保存输出形状
    print(json.dumps({"status": "ok", "last_shapes": last_shapes}, ensure_ascii=False, indent=2))  # 打印检查结果
    return 0  # 返回成功状态


if __name__ == "__main__":  # 判断是否为脚本直运行
    raise SystemExit(main())  # 以主函数返回码退出

