################################################################################
# FILE: run_combine_decoder_official.py
# PURPOSE: 使用官方 TransNASBench 模型实现（create_graph=False 路径）评估
#          多 proxy 聚合与真值的相关性，并便于对比 Graph 版实现。
#
# NOTE:
#   - 与 run_combine_decoder.py 不同，本脚本通过 create_graph=False + set_spec
#     让 NASLib 在 parse/set_op_indices 时调用 convert_op_indices_*_to_model，
#     进而使用原作者的 tnb101 create_model（与基准训练同构）。
#   - 提供 debug 打印开关，便于核查实例化的模型类型/decoder 结构。
#
# AUTHOR: ChatGPT (based on Enmin Lin's pipeline)
# DATE: 2025-01-XX
################################################################################

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import warnings
from tqdm import tqdm

# 抑制部分未来警告
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*squared.*")

# 路径设置
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
NASLIB_ROOT = ROOT_DIR / "NASLib"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(NASLIB_ROOT))

from naslib import utils as nas_utils
from proxy_TransNAS.utils.load_model import (
    load_transbench_classes,
    load_transbench_api,
    make_train_loader,
    truncate_loader,
    select_architectures_by_percentile,
    get_metric_name,
    set_op_indices_from_str,
)
from proxy_TransNAS.utils.rank_agg import nonlinear_ranking_aggregation
from proxy_TransNAS.proxies.factory import compute_proxy_score
from proxy_TransNAS.proxies.zico import get_loss_fn

get_train_val_loaders = nas_utils.get_train_val_loaders
compute_scores = nas_utils.compute_scores


# ============================================================================
# 辅助：模型构建与调试打印
# ============================================================================

def build_search_space(ss_name: str, task: str, args):
    """根据模式返回搜索空间实例（create_graph 开关由 args.model_impl 控制）。"""
    TransBench101SearchSpaceMicro, TransBench101SearchSpaceMacro, _ = load_transbench_classes()
    create_graph = args.model_impl == "graph"  # official 路径使用 False

    if ss_name == "micro":
        if task == "segmentsemantic":
            return TransBench101SearchSpaceMicro(dataset=task, create_graph=create_graph, n_classes=17)
        return TransBench101SearchSpaceMicro(dataset=task, create_graph=create_graph)
    else:
        return TransBench101SearchSpaceMacro(dataset=task, create_graph=create_graph)


def debug_model(graph, model, arch_str, op_indices, args, idx):
    """
    在关键节点打印模型信息，便于核查是否走了官方 create_model 路径。
    仅在 --debug 打开且 idx < debug_limit 时执行，避免刷屏。
    """
    if not args.debug or idx >= args.debug_limit:
        return

    edge_op = graph.edges[1, 2]["op"]
    print(f"[DEBUG] #{idx} arch={arch_str}")
    print(f"        impl={args.model_impl}, edge_op={type(edge_op)}, model_type={type(model)}")
    print(f"        op_indices={op_indices}")

    # 尝试打印 decoder 关键信息
    decoder = None
    if hasattr(edge_op, "model"):
        decoder = getattr(edge_op.model, "decoder", None)
    if decoder is not None:
        try:
            if hasattr(decoder, "model"):
                print(f"        decoder(seq) modules={len(list(decoder.model))}")
            elif hasattr(decoder, "conv14"):
                print(f"        decoder.conv14 out_channels={decoder.conv14.out_channels}")
            else:
                print(f"        decoder type={type(decoder)}")
        except Exception as e:
            print(f"        decoder inspect failed: {e}")


# ============================================================================
# 核心评估逻辑（官方模型路径）
# ============================================================================

def evaluate_task(task: str, ss_name: str, args, arch_strings=None):
    """在单个任务上评估多 proxy 聚合对官方模型的相关性。"""
    TransBench101SearchSpaceMicro, TransBench101SearchSpaceMacro, graph_module = load_transbench_classes()
    Metric = graph_module.Metric

    data_root = Path(args.data_root).resolve()
    dataset_api = load_transbench_api(data_root, task)
    api = dataset_api["api"]

    if args.dry_run:
        train_batches = []
        loss_fn = None
    else:
        train_loader = make_train_loader(task, data_root, args.batch_size, args.seed)
        train_batches = truncate_loader(train_loader, args.maxbatch)
        loss_fn = get_loss_fn(task).to(args.device)

    # 架构列表
    assert arch_strings is not None and len(arch_strings) > 0, "arch_strings 不能为空"

    gt_scores = []
    proxy_scores_dict = {proxy: [] for proxy in args.proxies}
    arch_hashes = []
    start = time.time()

    for i, arch_str in enumerate(tqdm(arch_strings, desc=f"[{task}] 官方模型评估", unit="arch")):
        # dry_run 时不实例化模型，直接取 GT 并用 0 占位，避免 train_batches 为空导致的下标错误
        if args.dry_run:
            arch_hashes.append(arch_str)
            metric_name = get_metric_name(task)
            gt = api.get_single_metric(arch_str, task, metric_name, mode="final")
            gt_scores.append(gt)
            for proxy in args.proxies:
                proxy_scores_dict[proxy].append(0.0)
            continue

        graph = build_search_space(ss_name, task, args)
        graph = set_op_indices_from_str(ss_name, graph, arch_str)
        graph.parse()
        model = graph.to(args.device)

        current_proxy_scores = {}
        skip_architecture = False

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

        if not skip_architecture:
            debug_model(graph, model, arch_str, graph.get_op_indices(), args, i)

        model.cpu()
        del model
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        if skip_architecture:
            del graph
            torch.cuda.empty_cache()
            gc.collect()
            continue

        arch_hashes.append(arch_str)
        metric_name = get_metric_name(task)
        gt = api.get_single_metric(arch_str, task, metric_name, mode="final")
        gt_scores.append(gt)
        for proxy in args.proxies:
            proxy_scores_dict[proxy].append(current_proxy_scores[proxy])

        del graph
        torch.cuda.empty_cache()
        gc.collect()

    elapsed = time.time() - start

    if len(gt_scores) > 0 and not args.dry_run:
        proxy_scores_for_aggregation = {proxy.upper(): proxy_scores_dict[proxy] for proxy in args.proxies}
        aggregated_scores = nonlinear_ranking_aggregation(proxy_scores_for_aggregation)
    else:
        aggregated_scores = [0.0] * len(arch_hashes)

    corr = compute_scores(ytest=gt_scores, test_pred=aggregated_scores)

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
        "proxy_scores": proxy_scores_dict,
        "individual_corrs": individual_corrs,
    }

    return result


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run official-model Multi-Proxy Ensemble on TransNAS-Bench-101")
    parser.add_argument("--tasks", nargs="+", default=["autoencoder", "segmentsemantic", "normal"], help="任务列表")
    parser.add_argument("--search_space", choices=["micro", "macro"], default="micro", help="搜索空间类型")
    parser.add_argument("--proxies", nargs="+", choices=["zico", "naswot", "flops", "swap", "zico_swap", "lswag", "fisher"],
                        default=["zico", "naswot", "swap", "flops"],
                        help="选择使用的 proxy 组合")
    parser.add_argument("--decoder_only", action="store_true", help="仅计算 decoder 部分")
    parser.add_argument("--data_root", type=str, default=str(NASLIB_ROOT / "data"), help="数据根路径")
    parser.add_argument("--batch_size", type=int, default=128, help="DataLoader batch 大小")
    parser.add_argument("--maxbatch", type=int, default=2, help="截断 batch 数")
    parser.add_argument("--num_samples", type=int, default=20, help="随机采样数量（sample_mode=random）")
    parser.add_argument("--start_percent", type=float, default=0.0, help="百分比分段起点（sample_mode=percent）")
    parser.add_argument("--end_percent", type=float, default=10.0, help="百分比分段终点（sample_mode=percent）")
    parser.add_argument("--flops_csv", type=str, default=str(ROOT_DIR / "proxy_TransNAS" / "flops_lookup" / "flops_macro_autoencoder.csv"),
                        help="flops lookup CSV（sample_mode=flops）")
    parser.add_argument("--start_arch_str", type=str, default=None, help="flops CSV 起始架构串（sample_mode=flops）")
    parser.add_argument("--arch_count", type=int, default=10, help="flops CSV 采样数量（sample_mode=flops）")
    parser.add_argument("--sample_mode", choices=["random", "percent", "flops"], default="random",
                        help="采样模式")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--dry_run", action="store_true", help="仅跑采样/GT，不计算 proxy")
    parser.add_argument("--model_impl", choices=["official", "graph"], default="official",
                        help="官方 create_model 路径或 Graph 版对比")
    parser.add_argument("--debug", action="store_true", help="打印模型类型/decoder 结构用于对齐调试")
    parser.add_argument("--debug_limit", type=int, default=2, help="调试打印的最大架构数")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data_root = Path(args.data_root).resolve()
    print("=" * 80)
    print(f"Sample mode: {args.sample_mode}")
    print(f"Search space: {args.search_space}")
    print(f"Tasks: {args.tasks}")
    print(f"Proxies: {args.proxies}")
    print(f"Model impl: {args.model_impl}")
    print("=" * 80)

    results = []
    for task in args.tasks:
        dataset_api = load_transbench_api(data_root, task)
        api = dataset_api["api"]

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
        else:  # flops
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

        print(f"\n[{task}] 结果对比:")
        for proxy in args.proxies:
            if proxy in res["individual_corrs"]:
                corr = res["individual_corrs"][proxy]
                proxy_name = proxy.upper()
                print(f"  {proxy_name:12} only: τ={corr['kendalltau']:.4f}, ρ={corr['spearman']:.4f}")
        proxy_names = "+".join([p.upper() for p in args.proxies])
        print(f"  {proxy_names:12}     : τ={res['kendalltau']:.4f}, ρ={res['spearman']:.4f}")
        print(f"  用时: {res['time']:.1f}s\n")

    print("=" * 80)
    print("=== 汇总结果 ===")
    print("=" * 80)
    proxy_names = "+".join([p.upper() for p in args.proxies])

    for res in results:
        print(f"\n{res['task']}:")
        best_single_tau = 0.0
        for proxy in args.proxies:
            if proxy in res["individual_corrs"]:
                corr = res["individual_corrs"][proxy]
                proxy_name = proxy.upper()
                print(f"  {proxy_name:12} only: τ={corr['kendalltau']:.4f}, ρ={corr['spearman']:.4f}")
                if corr["kendalltau"] is not None and not np.isnan(corr["kendalltau"]):
                    best_single_tau = max(best_single_tau, corr["kendalltau"])

        print(f"  {proxy_names:12}     : τ={res['kendalltau']:.4f}, ρ={res['spearman']:.4f}")
        improvement = res["kendalltau"] - best_single_tau
        print(f"  改进: τ {improvement:+.4f}")


if __name__ == "__main__":
    main()


