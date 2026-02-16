"""Strict Fairness 采样器。"""  # 定义模块用途

import argparse  # 导入参数解析模块
import json  # 导入JSON模块
import random  # 导入随机模块
from typing import Dict, List  # 导入类型注解


def generate_fair_cycle(rng: random.Random, num_blocks: int = 9) -> List[List[int]]:  # 定义公平周期生成函数
    """生成一个公平周期的3个架构编码。"""  # 说明函数用途
    block_permutations: List[List[int]] = []  # 初始化每个block的排列列表
    for _ in range(num_blocks):  # 按block数量循环生成
        options = [0, 1, 2]  # 初始化选项列表
        rng.shuffle(options)  # 原地打乱选项顺序
        block_permutations.append(options)  # 保存该block的排列
    cycle_codes: List[List[int]] = []  # 初始化周期编码列表
    for path_idx in range(3):  # 生成3条路径编码
        arch_code = []  # 初始化单条路径编码
        for block_idx in range(num_blocks):  # 遍历每个block
            arch_code.append(block_permutations[block_idx][path_idx])  # 取对应排列位置的选项
        cycle_codes.append(arch_code)  # 保存单条路径编码
    return cycle_codes  # 返回周期编码集合


def run_cycles(cycles: int, seed: int, num_blocks: int = 9) -> Dict:  # 定义多周期运行函数
    """运行多个公平周期并统计计数。"""  # 说明函数用途
    rng = random.Random(seed)  # 创建可复现随机数生成器
    counts = {block_idx: {0: 0, 1: 0, 2: 0} for block_idx in range(num_blocks)}  # 初始化计数字典
    records: List[List[List[int]]] = []  # 初始化周期记录列表
    for _ in range(cycles):  # 按周期数循环
        cycle_codes = generate_fair_cycle(rng=rng, num_blocks=num_blocks)  # 生成单个公平周期
        records.append(cycle_codes)  # 保存周期编码记录
        for arch_code in cycle_codes:  # 遍历周期内每条路径编码
            for block_idx, option in enumerate(arch_code):  # 遍历每个block选项
                counts[block_idx][int(option)] += 1  # 累加对应选项计数
    fairness_gap = 0  # 初始化公平差距
    for block_idx in range(num_blocks):  # 遍历每个block
        values = list(counts[block_idx].values())  # 读取该block三选项计数
        fairness_gap = max(fairness_gap, max(values) - min(values))  # 更新全局最大差距
    return {"counts": counts, "fairness_gap": fairness_gap, "records": records}  # 返回统计结果


def _build_parser() -> argparse.ArgumentParser:  # 定义参数解析器构建函数
    """创建命令行参数解析器。"""  # 说明函数用途
    parser = argparse.ArgumentParser(description="strict fairness sampler")  # 创建解析器
    parser.add_argument("--cycles", type=int, default=1, help="number of fairness cycles")  # 添加周期参数
    parser.add_argument("--seed", type=int, default=42, help="random seed")  # 添加种子参数
    parser.add_argument("--check", action="store_true", help="enable fairness assertion")  # 添加公平检查开关
    return parser  # 返回解析器


def main() -> int:  # 定义主函数
    """执行命令行入口。"""  # 说明函数用途
    parser = _build_parser()  # 构建参数解析器
    args = parser.parse_args()  # 解析命令行参数
    result = run_cycles(cycles=args.cycles, seed=args.seed)  # 运行采样统计
    if args.check and result["fairness_gap"] != 0:  # 判断检查模式下是否公平失败
        print(json.dumps(result, ensure_ascii=False, indent=2))  # 输出完整统计信息
        return 1  # 返回失败状态码
    summary = {  # 组织摘要输出
        "cycles": int(args.cycles),  # 写入周期数
        "seed": int(args.seed),  # 写入随机种子
        "fairness_gap": int(result["fairness_gap"]),  # 写入公平差距
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))  # 输出摘要结果
    return 0  # 返回成功状态


if __name__ == "__main__":  # 判断是否为脚本直运行
    raise SystemExit(main())  # 以主函数返回码退出

