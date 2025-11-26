"""
full_run_transnas.py  # 文件说明
批量运行 run_zico_transnas 的脚本，支持大采样或全空间评估，并记录日志。  # 功能描述
"""

import argparse  # 解析命令行参数
import json  # 读写 JSON 文件
import sys  # 操作 sys.path
import time  # 统计用时
from pathlib import Path  # 处理路径
from datetime import datetime  # 生成时间戳
import importlib.util  # 动态加载模块

import torch  # 判断设备
import numpy as np  # 预留可能的数值操作

# -----------------------------------------------------------------------------
# 常量：路径与模块加载
# -----------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent  # 当前脚本目录
RUN_ZICO_PATH = CURRENT_DIR / "run_zico_transnas.py"  # run_zico 脚本路径

spec = importlib.util.spec_from_file_location("run_zico_module", RUN_ZICO_PATH)  # 构造模块规范
run_zico = importlib.util.module_from_spec(spec)  # 创建模块对象
spec.loader.exec_module(run_zico)  # 加载 run_zico_transnas.py
evaluate_task = run_zico.evaluate_task  # 引用 evaluate_task 函数

# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)  # 若不存在则递归创建
    return path  # 返回路径对象


def save_json(obj, path: Path):
    with path.open("w", encoding="utf-8") as f:  # 打开文件写入
        json.dump(obj, f, indent=2, ensure_ascii=False)  # 序列化为 JSON


def make_args_namespace(base_args, num_samples, seed):
    """构造传给 evaluate_task 的 Namespace"""  # 函数说明
    return argparse.Namespace(  # 返回新的参数对象
        data_root=base_args.data_root,  # 数据根目录
        batch_size=base_args.batch_size,  # batch 大小
        maxbatch=base_args.maxbatch,  # 截断 batch 数
        device=base_args.device,  # 运行设备
        dry_run=base_args.dry_run,  # 是否 dry run
        num_samples=num_samples,  # 当前分块采样数
        seed=seed,  # 当前分块种子
        search_space=base_args.search_space,  # 使用的搜索空间
    )

# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
def run_full_experiment(args):
    """执行完整批量实验"""  # 函数注释
    log_root = ensure_dir(Path(args.log_root))  # 确保日志根目录存在
    exp_name = args.exp_name or datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成实验名
    exp_dir = ensure_dir(log_root / exp_name)  # 创建实验目录

    config = {  # 记录实验配置
        "tasks": args.tasks,  # 任务列表
        "search_space": args.search_space,  # 搜索空间
        "total_samples": args.total_samples,  # 总评估数量
        "chunk_size": args.chunk_size,  # 分块大小
        "batch_size": args.batch_size,  # batch 大小
        "maxbatch": args.maxbatch,  # 截断 batch 数
        "seed": args.seed,  # 基础随机种子
        "device": args.device,  # 运行设备
        "data_root": args.data_root,  # 数据目录
        "dry_run": args.dry_run,  # 是否 dry run
        "aggregate_task": args.aggregate_task,  # 聚合任务
        "created_at": datetime.now().isoformat(),  # 创建时间
    }
    save_json(config, exp_dir / "config.json")  # 保存配置文件

    all_chunks = []  # 存储所有分块日志
    remaining = args.total_samples  # 剩余待评估数量
    chunk_id = 0  # 分块计数器
    global_seed = args.seed  # 基础随机种子

    while remaining > 0:  # 逐块执行直到完成
        chunk_id += 1  # 当前分块编号
        num_samples = min(args.chunk_size, remaining)  # 当前分块采样数
        chunk_seed = global_seed + chunk_id  # 为分块生成新种子

        chunk_args = make_args_namespace(args, num_samples=num_samples, seed=chunk_seed)  # 生成分块参数
        chunk_results = []  # 存储分块内各任务结果

        start = time.time()  # 记录分块开始时间
        for task in args.tasks:  # 依次评估每个任务
            res = evaluate_task(task, args.search_space, chunk_args)  # 调用 run_zico 评估
            chunk_results.append(res)  # 保存结果
        elapsed = time.time() - start  # 计算分块耗时

        chunk_log = {  # 构造分块日志
            "chunk_id": chunk_id,  # 分块编号
            "num_samples": num_samples,  # 本块采样数
            "seed": chunk_seed,  # 本块随机种子
            "elapsed_sec": elapsed,  # 本块耗时
            "results": chunk_results,  # 本块结果
        }
        all_chunks.append(chunk_log)  # 汇总到总列表

        chunk_path = exp_dir / f"chunk_{chunk_id:03d}.json"  # 分块日志路径
        save_json(chunk_log, chunk_path)  # 写出分块日志

        remaining -= num_samples  # 更新剩余数量
        print(f"[Chunk {chunk_id}] 完成：num_samples={num_samples}, 用时={elapsed:.1f}s")  # 打印状态

    # --------------------- 聚合与总结输出 ---------------------
    aggregate_mode = (args.aggregate_task or "all").lower()  # 聚合任务模式
    all_gt_scores = []  # 聚合后的 GT 列表
    all_zico_scores = []  # 聚合后的 ZiCo 列表
    total_elapsed = 0.0  # 总耗时累积

    for chunk in all_chunks:  # 遍历所有分块
        total_elapsed += chunk["elapsed_sec"]  # 累加耗时
        for task_res in chunk["results"]:  # 遍历分块内各任务
            should_aggregate = aggregate_mode == "all" or task_res["task"].lower() == aggregate_mode  # 判断是否聚合
            if should_aggregate:  # 满足条件则聚合
                all_gt_scores.extend(task_res["gt"])  # 汇总 GT
                all_zico_scores.extend(task_res["pred"])  # 汇总 ZiCo

    if all_gt_scores and not args.dry_run:  # 若有数据且不是 dry run
        final_corr = run_zico.compute_scores(ytest=all_gt_scores, test_pred=all_zico_scores)  # 计算最终相关性
    else:
        final_corr = {"kendalltau": None, "spearman": None}  # 无法计算则返回空值

    kt = final_corr.get("kendalltau")  # 取 KendallTau
    sp = final_corr.get("spearman")  # 取 Spearman
    kt_str = f"{kt:.4f}" if isinstance(kt, (int, float)) else "N/A"  # 格式化显示
    sp_str = f"{sp:.4f}" if isinstance(sp, (int, float)) else "N/A"  # 格式化显示

    summary = {  # 构造汇总信息
        "config": config,  # 记录配置
        "num_chunks": len(all_chunks),  # 分块数量
        "chunks": [f"chunk_{c['chunk_id']:03d}.json" for c in all_chunks],  # 分块文件列表
        "total_samples_evaluated": len(all_gt_scores),  # 聚合的样本数量
        "total_elapsed_sec": total_elapsed,  # 总耗时
        "final_aggregated_corr": final_corr,  # 聚合相关性
    }
    save_json(summary, exp_dir / "summary.json")  # 写入 summary

    print(f"\n[SUMMARY] 聚合 {len(all_gt_scores)} 个样本的最终相关性：")  # 打印标题
    print(f"KendallTau: {kt_str}, Spearman: {sp_str}")  # 打印相关性
    print(f"总耗时：{total_elapsed:.1f}s")  # 打印耗时
    print(f"=== 全部完成，日志位于 {exp_dir} ===")  # 完成提示

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    """解析命令行参数"""  # 函数注释
    parser = argparse.ArgumentParser(description="Full runner for TransNAS ZiCo experiments")  # 创建解析器
    parser.add_argument("--tasks", nargs="+", default=["autoencoder", "segmentsemantic", "normal"], help="任务列表")  # 任务参数
    parser.add_argument("--search_space", choices=["micro", "macro"], default="micro", help="搜索空间类型")  # 搜索空间
    parser.add_argument("--total_samples", type=int, default=200, help="总共要评估多少个架构（分块执行）")  # 总采样数
    parser.add_argument("--chunk_size", type=int, default=20, help="每个分块一次评估多少个架构")  # 分块大小
    parser.add_argument("--batch_size", type=int, default=128, help="DataLoader 的 batch 大小")  # batch 大小
    parser.add_argument("--maxbatch", type=int, default=2, help="截断的 batch 数")  # 截断 batch 数
    parser.add_argument("--seed", type=int, default=42, help="基础随机种子")  # 随机种子
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="运行设备")  # 设备
    parser.add_argument("--dry_run", action="store_true", help="仅跑数据管线，不计算 ZiCo")  # dry run 开关
    parser.add_argument("--data_root", type=str, default=str(CURRENT_DIR.parent / "NASLib" / "data"), help="数据根目录")  # 数据路径
    parser.add_argument("--log_root", type=str, default=str(CURRENT_DIR / "full_run_logs"), help="日志根目录（自动创建）")  # 日志目录
    parser.add_argument("--exp_name", type=str, default=None, help="自定义实验子目录名称，不填则用时间戳")  # 实验名
    parser.add_argument("--aggregate_task", type=str, default="all", help="聚合相关性所使用的任务名，all 表示全部")  # 聚合任务
    return parser.parse_args()  # 返回解析结果


if __name__ == "__main__":  # 入口判断
    args = parse_args()  # 解析参数

    if args.total_samples <= 0:  # 检查总采样数
        raise ValueError("total_samples 必须 > 0")  # 抛出错误
    if args.chunk_size <= 0:  # 检查分块大小
        raise ValueError("chunk_size 必须 > 0")  # 抛出错误

    run_full_experiment(args)  # 运行主流程