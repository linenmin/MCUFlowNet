"""验证子网池构建器。"""  # 定义模块用途

import argparse  # 导入参数解析模块
import json  # 导入JSON模块
import random  # 导入随机模块
from typing import Dict, List  # 导入类型注解

BILINEAR_BASELINE_ARCH_CODE = [0, 0, 0, 0, 2, 1, 2, 2, 2]  # 0/1/2=3x3/5x5/7x7


def _seed_codes(num_blocks: int) -> List[List[int]]:  # 定义种子编码函数
    """构造覆盖性更强的基础编码集合。"""  # 说明函数用途
    codes = []  # 初始化编码列表
    codes.append([0 for _ in range(num_blocks)])  # 添加全0编码
    codes.append([1 for _ in range(num_blocks)])  # 添加全1编码
    codes.append([2 for _ in range(num_blocks)])  # 添加全2编码
    codes.append([(idx % 3) for idx in range(num_blocks)])  # 添加顺序轮转编码
    codes.append([((2 - idx) % 3) for idx in range(num_blocks)])  # 添加逆序轮转编码
    codes.append([((idx + 1) % 3) for idx in range(num_blocks)])  # 添加偏移轮转编码
    if num_blocks == 9:
        # 在前六个固定编码后，追加一个与 bilinear 基准对齐的固定子网。
        codes.append([int(item) for item in BILINEAR_BASELINE_ARCH_CODE])
    return codes  # 返回基础编码集合


def check_eval_pool_coverage(pool: List[List[int]], num_blocks: int = 9) -> Dict[str, object]:  # 定义覆盖检查函数
    """检查验证池是否覆盖每个block的三个选项。"""  # 说明函数用途
    counts = {block_idx: {0: 0, 1: 0, 2: 0} for block_idx in range(num_blocks)}  # 初始化计数字典
    for arch_code in pool:  # 遍历每个架构编码
        for block_idx, option in enumerate(arch_code):  # 遍历每个block选项
            counts[block_idx][int(option)] += 1  # 累加选项计数
    missing = []  # 初始化缺失列表
    for block_idx in range(num_blocks):  # 遍历每个block
        for option in (0, 1, 2):  # 遍历三个选项
            if counts[block_idx][option] <= 0:  # 判断选项是否缺失
                missing.append({"block": block_idx, "option": option})  # 记录缺失项
    return {"ok": len(missing) == 0, "counts": counts, "missing": missing}  # 返回覆盖检查结果


def build_eval_pool(seed: int, size: int, num_blocks: int = 9) -> List[List[int]]:  # 定义验证池构建函数
    """构建固定大小且满足覆盖约束的验证池。"""  # 说明函数用途
    if size < 3:  # 判断池大小是否过小
        raise ValueError("eval pool size must be >= 3")  # 抛出大小错误
    rng = random.Random(seed)  # 创建可复现随机数生成器
    pool: List[List[int]] = []  # 初始化子网池列表
    for code in _seed_codes(num_blocks=num_blocks):  # 遍历基础编码集合
        if len(pool) >= size:  # 判断是否达到目标池大小
            break  # 达到上限后退出循环
        pool.append(code)  # 添加基础编码到子网池
    while len(pool) < size:  # 判断子网池是否仍未填满
        candidate = [rng.randint(0, 2) for _ in range(num_blocks)]  # 生成随机候选编码
        if candidate in pool:  # 判断候选编码是否重复
            continue  # 重复时跳过当前候选
        pool.append(candidate)  # 添加新候选编码
    return pool  # 返回构建完成的子网池


def _build_parser() -> argparse.ArgumentParser:  # 定义参数解析器构建函数
    """创建命令行参数解析器。"""  # 说明函数用途
    parser = argparse.ArgumentParser(description="eval pool builder")  # 创建解析器
    parser.add_argument("--seed", type=int, default=42, help="random seed")  # 添加随机种子参数
    parser.add_argument("--size", type=int, default=12, help="eval pool size")  # 添加池大小参数
    parser.add_argument("--check", action="store_true", help="run coverage check")  # 添加覆盖检查开关
    return parser  # 返回解析器


def main() -> int:  # 定义主函数
    """执行命令行入口。"""  # 说明函数用途
    parser = _build_parser()  # 构建参数解析器
    args = parser.parse_args()  # 解析命令行参数
    pool = build_eval_pool(seed=args.seed, size=args.size)  # 构建验证子网池
    payload: Dict[str, object] = {"pool_size": len(pool), "pool": pool}  # 初始化输出字典
    if args.check:  # 判断是否开启覆盖检查
        payload["coverage"] = check_eval_pool_coverage(pool=pool)  # 写入覆盖检查结果
    print(json.dumps(payload, ensure_ascii=False, indent=2))  # 打印输出结果
    if args.check and not payload["coverage"]["ok"]:  # 判断检查模式下是否覆盖失败
        return 1  # 返回失败状态码
    return 0  # 返回成功状态码


if __name__ == "__main__":  # 判断是否为脚本直运行
    raise SystemExit(main())  # 以主函数返回码退出
