################################################################################
# FILE: run_combine_constraint.py
# PURPOSE: Evaluate multi-proxy ensemble on percentile-based architecture sets.
#          
# FEATURE: Select architectures by Ground Truth performance percentile range.
#          Each task selects its own architectures based on its GT distribution.
#          
#          Examples:
#          - [0%, 10%]: Top 10% best performing architectures
#          - [30%, 50%]: Middle 20% architectures  
#          - [90%, 100%]: Bottom 10% worst performing architectures
#          
#          Uses non-linear ranking aggregation to combine multiple zero-shot 
#          proxies (ZiCo, NASWOT, FLOPs) and evaluate their correlation with GT.
# 
# AUTHOR: Enmin Lin / KU Leuven
# CREATION DATE: December 4, 2025
# VERSION: 2.0
################################################################################

import argparse  # 解析命令行参数
import os  # 处理路径
import random  # 随机数控制
import sys  # 修改 sys.path 以复用已有代码
import time  # 计时
from pathlib import Path  # 路径对象
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
sys.path.insert(0, str(ROOT_DIR))  # 将项目根加入路径，支持 proxy_TransNAS.* 导入
sys.path.insert(0, str(NASLIB_ROOT))  # 导入 NASLib

from naslib import utils as nas_utils  # 避免与本地 utils 同名冲突
from proxy_TransNAS.utils.load_model import (  # 导入通用加载
    load_transbench_classes,  # 搜索空间类
    load_transbench_api,  # API 加载缓存
    make_train_loader,  # 构造 loader
    truncate_loader,  # 截断 loader
    select_architectures_by_percentile,  # 百分位选架构
    get_metric_name,  # metric 选择
    set_op_indices_from_str,  # 架构串解析
)  # 结束导入
from proxy_TransNAS.utils.rank_agg import nonlinear_ranking_aggregation  # 排名聚合
from proxy_TransNAS.proxies.factory import compute_proxy_score  # 统一 proxy 入口
from proxy_TransNAS.proxies.zico import get_loss_fn  # 损失选择

# 函数别名，便于调用
get_train_val_loaders = nas_utils.get_train_val_loaders  # 数据加载
compute_scores = nas_utils.compute_scores  # 相关性计算


# ============================================================================
# ZiCo 核心计算逻辑（从 run_zico_transnas.py 复用）
# ============================================================================


# ============================================================================
# 主评估逻辑
# ============================================================================

def evaluate_task(task: str, ss_name: str, args, arch_identifiers=None):
    """在单个任务上评估多 proxy 聚合与真值的相关性。"""
    # 动态取 Metric 与搜索空间类
    TransBench101SearchSpaceMicro, TransBench101SearchSpaceMacro, graph_module = load_transbench_classes()
    Metric = graph_module.Metric
    
    # 数据与 API 准备
    data_root = Path(args.data_root).resolve()
    dataset_api = load_transbench_api(data_root, task)
    api = dataset_api["api"]
    
    # 加载真实数据
    if args.dry_run:
        train_batches = []
        loss_fn = None
    else:
        train_loader = make_train_loader(task, data_root, args.batch_size, args.seed)
        train_batches = truncate_loader(train_loader, args.maxbatch)
        loss_fn = get_loss_fn(task).to(args.device)
    
    # 搜索空间与架构采样
    if ss_name == "micro":
        if task == "segmentsemantic":
            ss = TransBench101SearchSpaceMicro(dataset=task, create_graph=True, n_classes=17)
        else:
            ss = TransBench101SearchSpaceMicro(dataset=task, create_graph=True)
    else:
        ss = TransBench101SearchSpaceMacro(dataset=task, create_graph=True)
    
    # arch_identifiers 现在直接是架构字符串列表（由上游选择逻辑给出）
    assert arch_identifiers is not None and len(arch_identifiers) > 0, "arch_identifiers 不能为空"
    
    gt_scores = []  # 存储真值
    proxy_scores_dict = {proxy: [] for proxy in args.proxies}  # 动态存储各 proxy 分数
    arch_strs = []  # 存储架构字符串
    start = time.time()
    
    # 逐个重建架构并评估
    for i, arch_str in enumerate(tqdm(arch_identifiers, desc=f"[{task}] 评估架构", unit="arch")):
        # 从架构字符串重建 graph
        graph = ss.clone()
        graph = set_op_indices_from_str(ss_name, graph, arch_str)
        
        if args.dry_run:
            arch_strs.append(arch_str)
            metric_name = get_metric_name(task)
            gt = api.get_single_metric(arch_str, task, metric_name, mode="final")
            gt_scores.append(gt)
            for proxy in args.proxies:
                proxy_scores_dict[proxy].append(0.0)
            del graph
            continue
        
        # === 实例化模型 ===
        graph.parse()
        model = graph.to(args.device)
        
        # === 存储本架构的 proxy 分数 ===
        current_proxy_scores = {}
        skip_architecture = False  # 标记是否跳过本架构
        
        # === 按需计算各个 proxy ===
        for proxy in args.proxies:
            try:
                score = compute_proxy_score(model, proxy, train_batches, loss_fn, args.device)
                current_proxy_scores[proxy] = score
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"!!! OOM ({proxy}) at sample {i}. Skipping...")
                    skip_architecture = True
                    break
                else:
                    raise e
            except Exception as e:
                print(f"!!! Error ({proxy}) at sample {i}: {e}. Skipping...")
                skip_architecture = True
                break
        
        # === 清理模型 ===
        model.cpu()
        del model
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # === 如果出现错误，跳过本架构 ===
        if skip_architecture:
            del graph
            torch.cuda.empty_cache()
            gc.collect()
            continue
        
        # === 记录数据 ===
        arch_strs.append(arch_str)
        # 使用 TransNASBenchAPI 直接查询 GT，避免由于字符串格式差异导致的 KeyError
        metric_name = get_metric_name(task)
        gt = api.get_single_metric(arch_str, task, metric_name, mode="final")
        gt_scores.append(gt)
        for proxy in args.proxies:
            proxy_scores_dict[proxy].append(current_proxy_scores[proxy])
        
        # === 清理图对象 ===
        del graph
        torch.cuda.empty_cache()
        gc.collect()
    
    elapsed = time.time() - start
    
    # === 非线性排名聚合 ===
    if len(gt_scores) > 0 and not args.dry_run:
        # 准备排名聚合的数据（proxy 名称需要大写）
        proxy_scores_for_aggregation = {
            proxy.upper(): proxy_scores_dict[proxy] 
            for proxy in args.proxies
        }
        aggregated_scores = nonlinear_ranking_aggregation(proxy_scores_for_aggregation, skip_constant=True)
    else:
        aggregated_scores = [0.0] * len(arch_strs)
    
    # 相关性评估
    corr = compute_scores(ytest=gt_scores, test_pred=aggregated_scores)
    
    # 同时计算单独 proxy 的相关性（用于对比）
    individual_corrs = {}
    for proxy in args.proxies:
        if len(proxy_scores_dict[proxy]) > 0:
            scores_array = np.array(proxy_scores_dict[proxy])
            # 检查是否为常数 proxy（方差为 0）
            score_std = np.std(scores_array)
            if score_std < 1e-10:
                # 常数 proxy，无法计算相关性
                individual_corrs[proxy] = {
                    "kendalltau": np.nan,
                    "spearman": np.nan,
                }
            else:
                corr_single = compute_scores(ytest=gt_scores, test_pred=proxy_scores_dict[proxy])
                individual_corrs[proxy] = {
                    "kendalltau": corr_single.get("kendalltau"),
                    "spearman": corr_single.get("spearman"),
                }
    
    result = {
        "task": task,
        "kendalltau": corr.get("kendalltau"),
        "spearman": corr.get("spearman"),
        "time": elapsed,
        "gt": gt_scores,
        "pred": aggregated_scores,
        "arch_str": arch_strs,
        "proxy_scores": proxy_scores_dict,  # 存储所有 proxy 的分数
        "individual_corrs": individual_corrs,  # 存储单独 proxy 的相关性
    }
    
    return result


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Run Multi-Proxy Ensemble on Constrained Architecture Sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tasks", nargs="+", default=["autoencoder", "segmentsemantic", "normal"], 
                        help="任务列表")
    parser.add_argument("--search_space", choices=["micro", "macro"], default="micro", 
                        help="搜索空间类型")
    parser.add_argument("--proxies", nargs="+", choices=["zico", "naswot", "flops"], 
                        default=["zico", "naswot"], 
                        help="选择使用的 proxy 组合")
    parser.add_argument("--start_percent", type=float, default=0.0,
                        help="起始百分比（0-100），从性能最好开始计，例如 0=最好，90=接近最差")
    parser.add_argument("--end_percent", type=float, default=10.0,
                        help="结束百分比（0-100），例如 10=前10%，50=前50%，100=全部")
    parser.add_argument("--data_root", type=str, default=str(NASLIB_ROOT / "data"), 
                        help="数据根路径")
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="DataLoader 的 batch 大小")
    parser.add_argument("--maxbatch", type=int, default=2, 
                        help="截断的 batch 数")
    parser.add_argument("--seed", type=int, default=42, 
                        help="随机种子")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="设备")
    parser.add_argument("--dry_run", action="store_true", 
                        help="仅跑采样管线，不计算 proxy")
    return parser.parse_args()


def main():
    """主入口，遍历任务并打印结果。"""
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # 参数验证
    if not (0 <= args.start_percent <= 100 and 0 <= args.end_percent <= 100):
        print("错误：百分比必须在 0-100 之间")
        return
    
    if args.start_percent >= args.end_percent:
        print("错误：start_percent 必须小于 end_percent")
        return
    
    # 加载基础类和 API
    TransBench101SearchSpaceMicro, TransBench101SearchSpaceMacro, graph_module = load_transbench_classes()
    data_root = Path(args.data_root).resolve()
    
    # 打印配置信息
    print("=" * 80)
    print(f"架构选择模式: 百分比区间 [{args.start_percent:.1f}%, {args.end_percent:.1f}%]")
    print(f"搜索空间: {args.search_space.upper()}")
    print(f"任务列表: {', '.join(args.tasks)}")
    print(f"Proxies: {', '.join([p.upper() for p in args.proxies])}")
    print("=" * 80)
    print(f"注意：不同任务的性能分布不同，因此每个任务独立选择架构\n")
    
    # 遍历任务，每个任务单独选择架构
    results = []
    for task in args.tasks:
        print(f"\n{'='*60}")
        print(f"==> 评估任务 {task} (GT {args.start_percent:.1f}%-{args.end_percent:.1f}%)")
        print(f"{'='*60}")
        
        # 为当前任务加载 API 和搜索空间
        dataset_api = load_transbench_api(data_root, task)
        
        if args.search_space == "micro":
            if task == "segmentsemantic":
                ss = TransBench101SearchSpaceMicro(dataset=task, create_graph=True, n_classes=17)
            else:
                ss = TransBench101SearchSpaceMicro(dataset=task, create_graph=True)
        else:
            ss = TransBench101SearchSpaceMacro(dataset=task, create_graph=True)
        
        # 根据百分比区间选择架构
        task_arch_identifiers = select_architectures_by_percentile(
            dataset_api, args.search_space, task, args.start_percent, args.end_percent
        )
        
        if len(task_arch_identifiers) == 0:
            print(f"警告：任务 {task} 没有选择到任何架构，跳过")
            continue
        
        # 评估该任务
        res = evaluate_task(task, args.search_space, args, task_arch_identifiers)
        results.append(res)
        
        # 打印该任务的详细结果
        print(f"\n[{task}] 结果对比:")
        for proxy in args.proxies:
            if proxy in res['individual_corrs']:
                corr = res['individual_corrs'][proxy]
                proxy_name = proxy.upper()
                tau_val = corr['kendalltau']
                rho_val = corr['spearman']
                # 处理 np.nan 的情况
                if np.isnan(tau_val) or np.isnan(rho_val):
                    print(f"  {proxy_name:12} only: τ=nan, ρ=nan (常数 proxy，无法计算相关性)")
                else:
                    print(f"  {proxy_name:12} only: τ={tau_val:.4f}, ρ={rho_val:.4f}")
        
        proxy_names = "+".join([p.upper() for p in args.proxies])
        print(f"  {proxy_names:12}     : τ={res['kendalltau']:.4f}, ρ={res['spearman']:.4f}")
        print(f"  用时: {res['time']:.1f}s")
    
    # 汇总输出
    print("=" * 80)
    print("=== 汇总结果 ===")
    print("=" * 80)
    proxy_names = "+".join([p.upper() for p in args.proxies])
    
    for res in results:
        print(f"\n{res['task']}:")
        
        # 打印每个单独 proxy 的结果
        best_single_tau = 0.0
        for proxy in args.proxies:
            if proxy in res['individual_corrs']:
                corr = res['individual_corrs'][proxy]
                proxy_name = proxy.upper()
                tau_val = corr['kendalltau']
                rho_val = corr['spearman']
                # 处理 np.nan 的情况
                if np.isnan(tau_val) or np.isnan(rho_val):
                    print(f"  {proxy_name:12} only: τ=nan, ρ=nan (常数 proxy，无法计算相关性)")
                else:
                    print(f"  {proxy_name:12} only: τ={tau_val:.4f}, ρ={rho_val:.4f}")
                    best_single_tau = max(best_single_tau, tau_val)
        
        # 打印聚合结果
        print(f"  {proxy_names:12}     : τ={res['kendalltau']:.4f}, ρ={res['spearman']:.4f}")
        
        # 打印改进程度
        improvement = res['kendalltau'] - best_single_tau
        print(f"  改进: τ {improvement:+.4f}")


if __name__ == "__main__":
    main()

