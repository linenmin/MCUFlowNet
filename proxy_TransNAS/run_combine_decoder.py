################################################################################
# FILE: run_ZICO_NASWOT_transnas.py
# PURPOSE: Evaluate the correlation between aggregated multi-proxy ensemble 
#          and Ground Truth performance on TransNAS-Bench-101.
#          
# MODE: Uses non-linear ranking aggregation from AZ-NAS paper to combine
#       multiple zero-shot proxies (ZiCo, NASWOT, FLOPs) for better ranking 
#       consistency. Users can select any combination via --proxies argument.
# 
# AUTHOR: Enmin Lin / KU Leuven
# CREATION DATE: December 3, 2025
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
import pandas as pd  # 读 flops CSV
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

from proxy_TransNAS.utils.load_model import (  # 导入通用加载函数
    load_transbench_classes,  # 搜索空间类加载
    load_transbench_api,  # API 加载缓存
    make_train_loader,  # 构造训练 loader
    truncate_loader,  # 截断 loader
    select_architectures_by_percentile,  # 百分位选架构
    get_metric_name,  # metric 名
    set_op_indices_from_str,  # 从架构串写入 op_indices
)  # 结束导入
from proxy_TransNAS.utils.rank_agg import nonlinear_ranking_aggregation  # 排名聚合
from proxy_TransNAS.proxies.factory import compute_proxy_score  # 统一 proxy 入口
from proxy_TransNAS.proxies.zico import get_loss_fn  # 损失选择

# 函数别名，便于调用
get_train_val_loaders = nas_utils.get_train_val_loaders  # 数据加载
compute_scores = nas_utils.compute_scores  # 相关性计算





# ============================================================================
# 主评估逻辑
# ============================================================================

def evaluate_task(task: str, ss_name: str, args, arch_strings=None):
    """在单个任务上评估 decoder-only 多 proxy 与真值的相关性。"""
    # 动态取 Metric 与搜索空间类
    TransBench101SearchSpaceMicro, TransBench101SearchSpaceMacro, graph_module = load_transbench_classes()
    Metric = graph_module.Metric
    
    # 数据与 API 准备
    data_root = Path(args.data_root).resolve()
    dataset_api = load_transbench_api(data_root, task)
    api = dataset_api["api"]  # 取出 API 对象
    
    # 加载真实数据
    if args.dry_run:
        train_batches = []
        loss_fn = None
    else:
        train_loader = make_train_loader(task, data_root, args.batch_size, args.seed)
        train_batches = truncate_loader(train_loader, args.maxbatch)
        loss_fn = get_loss_fn(task).to(args.device)
    
    # 搜索空间
    if ss_name == "micro":
        if task == "segmentsemantic":
            ss = TransBench101SearchSpaceMicro(dataset=task, create_graph=True, n_classes=17)
        else:
            ss = TransBench101SearchSpaceMicro(dataset=task, create_graph=True)
    else:
        ss = TransBench101SearchSpaceMacro(dataset=task, create_graph=True)
    
    # 架构列表必须提供
    assert arch_strings is not None and len(arch_strings) > 0, "arch_strings 不能为空"
    
    gt_scores = []  # 存储真值
    proxy_scores_dict = {proxy: [] for proxy in args.proxies}  # 动态存储各 proxy 分数
    arch_hashes = []  # 存储架构哈希
    start = time.time()
    
    # 逐个重建架构并评估
    for i, arch_str in enumerate(tqdm(arch_strings, desc=f"[{task}] 评估架构", unit="arch", leave=False)):
        # dry_run 时不实例化模型，直接取 GT 并用 0 占位，避免 train_batches 为空导致的下标错误
        if args.dry_run:
            arch_hashes.append(arch_str)
            metric_name = get_metric_name(task)
            gt = api.get_single_metric(arch_str, task, metric_name, mode="final")
            gt_scores.append(gt)
            for proxy in args.proxies:
                proxy_scores_dict[proxy].append(0.0)
            continue

        # 从架构串重建 graph
        graph = ss.clone()
        graph = set_op_indices_from_str(ss_name, graph, arch_str)
        
        # === 实例化模型 ===
        graph.parse()
        model = graph.to(args.device)
        
        # === 存储本架构的 proxy 分数 ===
        current_proxy_scores = {}
        skip_architecture = False  # 标记是否跳过本架构
        
        # === 按需计算各个 proxy ===
        for proxy in args.proxies:
            try:
                score = compute_proxy_score(model, proxy, train_batches, loss_fn, args.device, decoder_only=args.decoder_only)
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
        arch_hashes.append(arch_str)
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
        aggregated_scores = nonlinear_ranking_aggregation(proxy_scores_for_aggregation)
    else:
        aggregated_scores = [0.0] * len(arch_hashes)
    
    # 相关性评估
    corr = compute_scores(ytest=gt_scores, test_pred=aggregated_scores)
    
    # 同时计算单独 proxy 的相关性（用于对比）
    individual_corrs = {}
    for proxy in args.proxies:
        if len(proxy_scores_dict[proxy]) > 0:
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
        "arch_hash": arch_hashes,
        "proxy_scores": proxy_scores_dict,  # 存储所有 proxy 的分数
        "individual_corrs": individual_corrs,  # 存储单独 proxy 的相关性
    }
    
    return result


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run decoder-only Multi-Proxy Ensemble on TransNAS-Bench-101")
    parser.add_argument("--tasks", nargs="+", default=["autoencoder", "segmentsemantic", "normal"], help="任务列表")
    parser.add_argument("--search_space", choices=["micro", "macro"], default="micro", help="搜索空间类型")
    parser.add_argument("--proxies", nargs="+", choices=["zico", "naswot", "flops", "swap", "zico_swap", "lswag", "fisher", "loss"], 
                        default=["zico", "naswot", "swap", "flops"], 
                        help="选择使用的 proxy 组合（例如：--proxies zico naswot flops swap）")
    parser.add_argument("--decoder_only", action="store_true", help="仅计算 decoder 部分（不加则全模型）")
    parser.add_argument("--data_root", type=str, default=str(NASLIB_ROOT / "data"), help="数据根路径")
    parser.add_argument("--batch_size", type=int, default=128, help="DataLoader 的 batch 大小")
    parser.add_argument("--maxbatch", type=int, default=2, help="截断的 batch 数")
    parser.add_argument("--num_samples", type=int, default=20, help="随机采样数量（sample_mode=random 时有效）")
    parser.add_argument("--start_percent", type=float, default=0.0, help="按百分比分段采样起点（sample_mode=percent）")
    parser.add_argument("--end_percent", type=float, default=10.0, help="按百分比分段采样终点（sample_mode=percent）")
    parser.add_argument("--flops_csv", type=str, default=str(ROOT_DIR / "proxy_TransNAS" / "flops_lookup" / "flops_macro_autoencoder.csv"),
                        help="flops lookup CSV 路径（sample_mode=flops）")
    parser.add_argument("--start_arch_str", type=str, default=None, help="flops CSV 起始架构串（sample_mode=flops）")
    parser.add_argument("--arch_count", type=int, default=10, help="flops CSV 采样数量（sample_mode=flops）")
    parser.add_argument("--sample_mode", choices=["random", "percent", "flops"], default="random",
                        help="采样模式：random 随机数量；percent 按 GT 百分比；flops 按 flops CSV 顺序切片")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--dry_run", action="store_true", help="仅跑采样管线，不计算 proxy")
    return parser.parse_args()


def main():
    """主入口，遍历任务并打印结果。"""
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    TransBench101SearchSpaceMicro, TransBench101SearchSpaceMacro, graph_module = load_transbench_classes()
    data_root = Path(args.data_root).resolve()
    # 打印配置
    # print("=" * 80)
    # print(f"Sample mode: {args.sample_mode}")
    # print(f"Search space: {args.search_space}")
    # print(f"Tasks: {args.tasks}")
    # print(f"Proxies: {args.proxies}")
    # print("=" * 80)
    
    results = []
    for task in args.tasks:
        # 准备 API
        dataset_api = load_transbench_api(data_root, task)
        api = dataset_api["api"]
        
        # 按模式选择架构串
        if args.sample_mode == "random":
            arch_pool = api.all_arch_dict[args.search_space]
            if len(arch_pool) < args.num_samples:
                print(f"警告：可用架构 {len(arch_pool)} 少于请求数量 {args.num_samples}")
            arch_strings = random.sample(arch_pool, min(args.num_samples, len(arch_pool)))
        elif args.sample_mode == "percent":
            arch_strings = select_architectures_by_percentile(
                dataset_api, args.search_space, task, args.start_percent, args.end_percent
            )
            if len(arch_strings) == 0:
                print(f"任务 {task} 在区间 {args.start_percent}-{args.end_percent}% 无架构，跳过")
                continue
        else:  # flops CSV
            csv_path = Path(args.flops_csv)
            assert csv_path.exists(), f"CSV 不存在: {csv_path}"
            df = pd.read_csv(csv_path).sort_values(by="flops").reset_index(drop=True)
            if args.start_arch_str is None:
                start_idx = 0
            else:
                hit = df.index[df["arch_str"] == args.start_arch_str].tolist()
                assert len(hit) > 0, f"start_arch_str 未在 CSV 中找到: {args.start_arch_str}"
                start_idx = hit[0]
            end_idx = start_idx + args.arch_count
            arch_strings = df.iloc[start_idx:end_idx]["arch_str"].tolist()
            if len(arch_strings) == 0:
                print(f"任务 {task} 在 CSV 切片为空，跳过")
                continue
        
        print(f"==> 评估任务 {task}, 架构数: {len(arch_strings)}")
        res = evaluate_task(task, args.search_space, args, arch_strings)
        results.append(res)
        
        # 打印详细结果（包含单独 proxy 的对比）
        print(f"\n[{task}] 结果对比:")
        for proxy in args.proxies:
            if proxy in res['individual_corrs']:
                corr = res['individual_corrs'][proxy]
                proxy_name = proxy.upper()
                print(f"  {proxy_name:12} only: τ={corr['kendalltau']:.4f}, ρ={corr['spearman']:.4f}")
        proxy_names = "+".join([p.upper() for p in args.proxies])
        print(f"  {proxy_names:12}     : τ={res['kendalltau']:.4f}, ρ={res['spearman']:.4f}")
        print(f"  用时: {res['time']:.1f}s\n")
    
    # 汇总输出
    print("=" * 20)
    print("=== Final Results ===")
    print("=" * 20)
    proxy_names = "+".join([p.upper() for p in args.proxies])
    
    for res in results:
        print(f"\n{res['task']}:")
        
        best_single_tau = 0.0
        for proxy in args.proxies:
            if proxy in res['individual_corrs']:
                corr = res['individual_corrs'][proxy]
                proxy_name = proxy.upper()
                print(f"  {proxy_name:12} only: τ={corr['kendalltau']:.4f}, ρ={corr['spearman']:.4f}")
                if corr['kendalltau'] is not None and not np.isnan(corr['kendalltau']):
                    best_single_tau = max(best_single_tau, corr['kendalltau'])
        
        print(f"  {proxy_names:12}     : τ={res['kendalltau']:.4f}, ρ={res['spearman']:.4f}")
        improvement = res['kendalltau'] - best_single_tau
        print(f"  改进: τ {improvement:+.4f}")


if __name__ == "__main__":
    main()

