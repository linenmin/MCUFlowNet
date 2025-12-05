################################################################################
# FILE: run_NASWOT_transnas.py
# PURPOSE: Evaluate the correlation between NASWOT (NAS Without Training) and 
#          Ground Truth performance on TransNAS-Bench-101.
#          
# MODE: This script operates in a **Sampling and Correlation Printing** mode. 
#       It randomly samples a small number of architectures, computes their NASWOT 
#       scores, retrieves their GT scores, and prints the resulting Kendall Tau 
#       and Spearman correlation coefficients.
# 
# AUTHOR: Enmin Lin / KU Leuven
# CREATION DATE: December 3, 2025
# VERSION: 1.0
################################################################################

import argparse  # 解析命令行参数
import os  # 处理路径
import random  # 随机数控制
import sys  # 修改 sys.path 以复用已有代码
import time  # 计时
from pathlib import Path  # 路径对象
import itertools  # 截断数据迭代
import importlib.util  # 动态加载模块

import torch  # 张量与设备管理
import numpy as np  # 数值计算
import warnings  # 警告控制
from tqdm import tqdm  # 进度条显示

# 忽略特定的 FutureWarning，保持输出整洁
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*squared.*")

# 插入 NASLib 的代码路径
CURRENT_DIR = Path(__file__).resolve().parent  # 本文件所在目录
ROOT_DIR = CURRENT_DIR.parent  # MCUFlowNet 根目录
NASLIB_ROOT = ROOT_DIR / "NASLib"  # NASLib 代码根目录
sys.path.insert(0, str(ROOT_DIR))  # 项目根路径，支持 proxy_TransNAS 包导入
sys.path.insert(0, str(NASLIB_ROOT))  # 导入 NASLib

from naslib import utils as nas_utils  # 避免与本地 utils 同名冲突
from proxy_TransNAS.utils.load_model import (  # 复用通用加载
    load_transbench_classes,  # 搜索空间类
    load_transbench_api,  # API 加载缓存
    make_train_loader,  # 构造 loader
    truncate_loader,  # 截断 loader
    sample_architecture_identifiers,  # 采样架构
)  # 结束导入
from proxy_TransNAS.proxies.naswot import compute_naswot_score  # NASWOT 计算

# 函数别名，便于调用
get_train_val_loaders = nas_utils.get_train_val_loaders  # 数据加载
compute_scores = nas_utils.compute_scores  # 相关性计算




# ============================================================================
# NASWOT 核心计算逻辑（改编自 compute_NASWOT_score.py）
# ============================================================================

def evaluate_task(task: str, ss_name: str, args, shared_arch_identifiers=None):
    """在单个任务上评估 NASWOT 与真值的相关性。"""
    # 动态取 Metric 与搜索空间类
    TransBench101SearchSpaceMicro, TransBench101SearchSpaceMacro, graph_module = load_transbench_classes()  # 取类和模块
    Metric = graph_module.Metric  # 直接使用模块中的 Metric，避免类型不一致
    
    # 数据与 API 准备
    data_root = Path(args.data_root).resolve()  # 数据路径
    dataset_api = load_transbench_api(data_root, task)  # 加载表格 API
    
    # 加载真实数据（与 ZiCo 完全一致）
    if args.dry_run:
        train_batches = []  # dry run 不需要数据
    else:
        train_loader = make_train_loader(task, data_root, args.batch_size, args.seed)  # 构造训练 loader
        train_batches = truncate_loader(train_loader, args.maxbatch)  # 截断 batch
    
    # 搜索空间与架构采样
    if ss_name == "micro":
        # 显式指定 n_classes 为 17 (segmentsemantic 的类别数)
        if task == "segmentsemantic":
            ss = TransBench101SearchSpaceMicro(dataset=task, create_graph=True, n_classes=17)
        else:
            ss = TransBench101SearchSpaceMicro(dataset=task, create_graph=True)
    else:
        ss = TransBench101SearchSpaceMacro(dataset=task, create_graph=True)
    
    # === 如果提供了共享架构列表，直接使用；否则自己采样（兼容性）===
    if shared_arch_identifiers is not None:
        arch_identifiers = shared_arch_identifiers  # 使用共享的采样结果
    else:
        arch_identifiers = sample_architecture_identifiers(ss, dataset_api, args.num_samples, args.seed)
    
    gt_scores = []  # 存储真值
    naswot_scores = []  # 存储 NASWOT
    arch_hashes = []  # 存储每个架构的哈希（op_indices）
    start = time.time()  # 计时开始
    
    # 逐个重建架构并评估
    for i, arch_identifier in enumerate(tqdm(arch_identifiers, desc=f"[{task}] 评估架构", unit="arch")):
        # 从标识符重建 graph
        graph = ss.clone()
        graph.set_op_indices(list(arch_identifier))  # 从 hash 恢复架构
        
        if args.dry_run:
            # dry_run 模式：记录所有架构
            arch_hashes.append(list(arch_identifier))
            gt = graph.query(metric=Metric.VAL_ACCURACY, dataset=task, dataset_api=dataset_api)
            gt_scores.append(gt)
            naswot_scores.append(0.0)  # 仅占位
            del graph  # 释放 graph
            continue  # 跳过计算
        
        # === 实例化模型并评估 ===
        graph.parse()  # 实例化模型参数
        model = graph.to(args.device)  # 取具体模型并放设备
        model.eval()  # 评估模式（NASWOT 不需要训练模式）
        
        # 计算 NASWOT 分数（先计算，成功后才记录数据）
        try:
            score = compute_naswot_score(model, train_batches, args.device)  # 计算 NASWOT
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"!!! OOM detected at sample {i}. Skipping this architecture...")
                torch.cuda.empty_cache()
                # 关键：OOM 时直接跳过，不记录任何数据
                model.cpu()
                del model
                del graph
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                continue  # 跳过，不记录 GT 和 NASWOT
            else:
                raise e
        except Exception as e:
            print(f"!!! Error at sample {i}: {e}. Skipping...")
            model.cpu()
            del model
            del graph
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            continue
        
        # 只有成功计算 NASWOT 后，才记录数据
        arch_hashes.append(list(arch_identifier))  # 记录架构哈希
        gt = graph.query(metric=Metric.VAL_ACCURACY, dataset=task, dataset_api=dataset_api)  # 查询真值
        gt_scores.append(gt)  # 记录真值
        naswot_scores.append(score)  # 记录分数
        
        # === 显存和内存清理（彻底释放） ===
        model.cpu()  # 关键：必须移回 CPU
        del model  # 删除模型引用
        del graph  # 删除 graph 引用（立即释放，不累积）
        
        torch.cuda.empty_cache()  # 清理显存碎片
        
        # === 强制 GC ===
        import gc
        gc.collect()
    
    elapsed = time.time() - start  # 总耗时
    
    # 相关性评估
    corr = compute_scores(ytest=gt_scores, test_pred=naswot_scores)  # 计算相关性
    return {
        "task": task,  # 任务名
        "kendalltau": corr.get("kendalltau"),  # Kendall Tau
        "spearman": corr.get("spearman"),  # Spearman
        "time": elapsed,  # 用时
        "gt": gt_scores,  # 真值列表（与架构顺序一致）
        "pred": naswot_scores,  # 预测列表（与架构顺序一致）
        "arch_hash": arch_hashes,  # 架构哈希列表（每个元素通常是长度为 6 的 op_indices 列表）
    }  # 返回结果字典


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run NASWOT on TransNAS-Bench-101")  # 创建解析器
    parser.add_argument("--tasks", nargs="+", default=["autoencoder", "segmentsemantic", "normal"], help="任务列表")  # 任务参数
    parser.add_argument("--search_space", choices=["micro", "macro"], default="micro", help="搜索空间类型")  # 空间选择
    parser.add_argument("--data_root", type=str, default=str(NASLIB_ROOT / "data"), help="数据根路径")  # 数据路径
    parser.add_argument("--num_samples", type=int, default=20, help="采样架构数量")  # 采样数量
    parser.add_argument("--batch_size", type=int, default=128, help="DataLoader 的 batch 大小")  # batch 大小
    parser.add_argument("--maxbatch", type=int, default=1, help="截断的 batch 数（NASWOT 只需要 1 个）")  # 截断 batch 数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")  # 种子
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")  # 设备选择
    parser.add_argument("--dry_run", action="store_true", help="仅跑采样管线，不计算 NASWOT")  # dry run
    return parser.parse_args()  # 返回参数


def main():
    """主入口，遍历任务并打印结果。"""
    args = parse_args()  # 解析参数
    torch.manual_seed(args.seed)  # 设定种子
    random.seed(args.seed)  # 设定种子
    
    # === 在任务循环之前，先采样一次架构（所有任务共享）===
    print("=" * 80)
    print("Step 0: 为所有任务采样架构（共享）")
    print("=" * 80)
    
    # 创建搜索空间（用于采样）
    TransBench101SearchSpaceMicro, TransBench101SearchSpaceMacro, graph_module = load_transbench_classes()
    data_root = Path(args.data_root).resolve()
    dataset_api = load_transbench_api(data_root, args.tasks[0])  # 用第一个任务获取 API（API 本身是任务无关的）
    
    # 创建搜索空间（用第一个任务，因为搜索空间结构是任务无关的）
    if args.search_space == "micro":
        # 显式指定 n_classes 为 17 (segmentsemantic 的类别数)
        if args.tasks[0] == "segmentsemantic":
            ss = TransBench101SearchSpaceMicro(dataset=args.tasks[0], create_graph=True, n_classes=17)
        else:
            ss = TransBench101SearchSpaceMicro(dataset=args.tasks[0], create_graph=True)
    else:
        ss = TransBench101SearchSpaceMacro(dataset=args.tasks[0], create_graph=True)
    
    shared_arch_identifiers = sample_architecture_identifiers(ss, dataset_api, args.num_samples, args.seed)
    print(f"✓ 采样完成，所有任务将共享这 {len(shared_arch_identifiers)} 个架构\n")
    
    # === 遍历任务，传入共享的架构列表 ===
    results = []  # 结果列表
    for task in args.tasks:
        print(f"==> 评估任务 {task}")  # 打印任务
        res = evaluate_task(task, args.search_space, args, shared_arch_identifiers)  # 传入共享列表
        results.append(res)  # 收集结果
        # 打印简要结果
        print(
            f"[{task}] kendalltau={res['kendalltau']:.4f} spearman={res['spearman']:.4f} time={res['time']:.1f}s"
        )  # 打印相关性与时间
    
    # 汇总输出
    print("=" * 80)
    print("=== 汇总结果 ===")  # 标题
    print("=" * 80)
    for res in results:
        print(
            f"{res['task']}: kendalltau={res['kendalltau']:.4f}, spearman={res['spearman']:.4f}, time={res['time']:.1f}s"
        )  # 每任务结果


if __name__ == "__main__":
    main()  # 运行主函数

