"""
analyze_full_run.py
分析 full_run_transnas.py 生成的日志文件夹，按任务去重架构并重新计算相关性指标。
"""

import argparse  # 解析命令行参数
import json  # 读取 JSON 文件
from pathlib import Path  # 处理路径
from collections import defaultdict  # 用于按任务分组数据
import numpy as np  # 数值计算
from scipy import stats  # 计算 Kendall Tau 和 Spearman


def load_config(exp_dir: Path):
    """加载实验配置文件"""
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_task_data(exp_dir: Path):
    """收集所有 chunk 文件中的任务数据，按任务分组"""
    task_data = defaultdict(lambda: {
        "arch_hashes": [],  # 架构哈希列表
        "gt_scores": [],    # 真值列表
        "proxy_scores": [], # Proxy 分数列表
    })
    
    # 遍历所有 chunk_*.json 文件
    for chunk_file in sorted(exp_dir.glob("chunk_*.json")):
        with chunk_file.open("r", encoding="utf-8") as f:
            chunk_data = json.load(f)
        
        task = chunk_data["task"]  # 任务名称
        result = chunk_data["result"]  # 结果字典
        
        # 提取数据
        arch_hashes = result.get("arch_hash", [])  # 架构哈希
        gt_scores = result.get("gt", [])  # 真值
        proxy_scores = result.get("pred", [])  # Proxy 分数
        
        # 累积到对应任务
        task_data[task]["arch_hashes"].extend(arch_hashes)
        task_data[task]["gt_scores"].extend(gt_scores)
        task_data[task]["proxy_scores"].extend(proxy_scores)
    
    return task_data


def deduplicate_by_arch(arch_hashes, gt_scores, proxy_scores):
    """根据架构哈希去重，保留第一次出现的架构"""
    seen = set()  # 已见过的架构哈希集合
    unique_arch_hashes = []  # 去重后的架构哈希
    unique_gt = []  # 去重后的真值
    unique_proxy = []  # 去重后的 Proxy 分数
    
    for arch_hash, gt, proxy in zip(arch_hashes, gt_scores, proxy_scores):
        # 将 list 转为 tuple 以便作为 set 的元素
        arch_tuple = tuple(arch_hash) if isinstance(arch_hash, list) else arch_hash
        
        if arch_tuple not in seen:  # 如果是新架构
            seen.add(arch_tuple)  # 标记为已见
            unique_arch_hashes.append(arch_hash)  # 保存架构
            unique_gt.append(gt)  # 保存真值
            unique_proxy.append(proxy)  # 保存 Proxy 分数
    
    return unique_arch_hashes, unique_gt, unique_proxy


def compute_correlation(gt_scores, proxy_scores):
    """计算 Kendall Tau 和 Spearman 相关性"""
    gt_array = np.array(gt_scores)  # 转为 NumPy 数组
    proxy_array = np.array(proxy_scores)  # 转为 NumPy 数组
    
    # 计算 Kendall Tau
    kendalltau, _ = stats.kendalltau(gt_array, proxy_array)
    
    # 计算 Spearman
    spearman, _ = stats.spearmanr(gt_array, proxy_array)
    
    return kendalltau, spearman


def analyze_experiment(exp_dir: Path):
    """分析单个实验文件夹"""
    exp_dir = Path(exp_dir)  # 转为 Path 对象
    
    if not exp_dir.exists():  # 检查路径是否存在
        print(f"错误：路径不存在 {exp_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"分析实验文件夹: {exp_dir.name}")
    print(f"{'='*80}\n")
    
    # 加载配置
    config = load_config(exp_dir)
    if config:
        print("实验配置:")
        print(f"  Proxy 方法: {config.get('proxy', 'N/A')}")
        print(f"  搜索空间: {config.get('search_space', 'N/A')}")
        print(f"  任务列表: {config.get('tasks', 'N/A')}")
        print(f"  总采样数: {config.get('total_samples', 'N/A')}")
        print(f"  分块大小: {config.get('chunk_size', 'N/A')}")
        print(f"  Batch 大小: {config.get('batch_size', 'N/A')}")
        print(f"  随机种子: {config.get('seed', 'N/A')}")
        print(f"  创建时间: {config.get('created_at', 'N/A')}")
        print()
    
    # 收集所有任务数据
    task_data = collect_task_data(exp_dir)
    
    if not task_data:  # 如果没有数据
        print("警告：未找到任何 chunk 文件")
        return
    
    # 按任务分析
    for task, data in sorted(task_data.items()):
        print(f"\n{'─'*80}")
        print(f"任务: {task}")
        print(f"{'─'*80}")
        
        arch_hashes = data["arch_hashes"]  # 所有架构哈希
        gt_scores = data["gt_scores"]  # 所有真值
        proxy_scores = data["proxy_scores"]  # 所有 Proxy 分数
        
        total_samples = len(arch_hashes)  # 总采样数
        print(f"  原始采样数: {total_samples}")
        
        # 去重
        unique_archs, unique_gt, unique_proxy = deduplicate_by_arch(
            arch_hashes, gt_scores, proxy_scores
        )
        
        unique_count = len(unique_archs)  # 去重后数量
        duplicate_count = total_samples - unique_count  # 重复数量
        
        print(f"  去重后架构数: {unique_count}")
        print(f"  重复架构数: {duplicate_count}")
        print(f"  重复率: {duplicate_count / total_samples * 100:.2f}%")
        
        # 计算相关性
        if unique_count >= 2:  # 至少需要 2 个样本才能计算相关性
            kendalltau, spearman = compute_correlation(unique_gt, unique_proxy)
            print(f"\n  去重后相关性指标:")
            print(f"    Kendall Tau: {kendalltau:.6f}")
            print(f"    Spearman:    {spearman:.6f}")
        else:
            print(f"\n  警告：样本数不足 (n={unique_count})，无法计算相关性")
        
        # 显示 GT 和 Proxy 的统计信息
        if unique_count > 0:
            print(f"\n  真值 (GT) 统计:")
            print(f"    均值: {np.mean(unique_gt):.6f}")
            print(f"    标准差: {np.std(unique_gt):.6f}")
            print(f"    最小值: {np.min(unique_gt):.6f}")
            print(f"    最大值: {np.max(unique_gt):.6f}")
            
            print(f"\n  Proxy 分数统计:")
            print(f"    均值: {np.mean(unique_proxy):.6f}")
            print(f"    标准差: {np.std(unique_proxy):.6f}")
            print(f"    最小值: {np.min(unique_proxy):.6f}")
            print(f"    最大值: {np.max(unique_proxy):.6f}")
    
    print(f"\n{'='*80}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="分析 full_run_transnas.py 生成的实验日志"
    )
    parser.add_argument(
        "exp_dir",
        type=str,
        help="实验文件夹路径，例如: full_run_logs/20251202_120054_myscore_micro"
    )
    
    args = parser.parse_args()
    analyze_experiment(args.exp_dir)


if __name__ == "__main__":
    main()

