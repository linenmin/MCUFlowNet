"""
full_run_transnas.py  # 文件说明
批量运行 run_zico_transnas 或 run_myscore_transnas 的脚本，支持多搜索空间、多代理方法顺序评估。  # 功能描述
"""

import argparse  # 解析命令行参数
import json  # 读写 JSON 文件
import sys  # 操作 sys.path
import time  # 统计用时
from pathlib import Path  # 处理路径
from datetime import datetime  # 生成时间戳
import importlib.util  # 动态加载模块
import gc  # 垃圾回收

import torch  # 判断设备
import numpy as np  # 预留可能的数值操作
from tqdm import tqdm  # 进度条显示

# -----------------------------------------------------------------------------
# 常量：路径与模块加载
# -----------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent  # 当前脚本目录

# 定义不同 Proxy 对应的脚本路径
PROXY_SCRIPTS = {
    "zico": CURRENT_DIR / "run_zico_transnas.py",  # ZiCo 脚本路径
    "myscore": CURRENT_DIR / "run_myscore_transnas.py",  # MyScore (C-SWAG) 脚本路径
}

def load_proxy_module(proxy_name):
    """动态加载指定 Proxy 的评估模块"""  # 函数说明
    script_path = PROXY_SCRIPTS.get(proxy_name)
    if not script_path or not script_path.exists():
        raise ValueError(f"未找到 Proxy 脚本: {proxy_name}")
        
    spec = importlib.util.spec_from_file_location(f"run_{proxy_name}_module", script_path)  # 构造模块规范
    module = importlib.util.module_from_spec(spec)  # 创建模块对象
    spec.loader.exec_module(module)  # 加载脚本
    return module.evaluate_task, module  # 返回评估函数和模块对象

# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)  # 若不存在则递归创建
    return path  # 返回路径对象


# 新增：自定义 JSON Encoder 以处理 NumPy 类型 (如 int32)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        # 使用抽象基类检查，兼容 NumPy 1.x 和 2.x
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_json(obj, path: Path):
    with path.open("w", encoding="utf-8") as f:  # 打开文件写入
        # 使用 cls=NumpyEncoder 解决 "Object of type int32 is not JSON serializable" 问题
        json.dump(obj, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)  # 序列化为 JSON


def make_args_namespace(base_args, num_samples, seed):
    """构造传给 evaluate_task 的 Namespace"""  # 函数说明
    # 获取可选参数（如果 base_args 中有的话，否则使用默认值）
    top_k_percent = getattr(base_args, 'top_k_percent', 0.50)  # Top-K 截断比例（默认 0.50）
    alpha_threshold = getattr(base_args, 'alpha_threshold', 0.0)  # 关键层筛选阈值（默认 0.0）
    
    return argparse.Namespace(  # 返回新的参数对象
        data_root=base_args.data_root,  # 数据根目录
        batch_size=base_args.batch_size,  # batch 大小
        maxbatch=base_args.maxbatch,  # 截断 batch 数
        device=base_args.device,  # 运行设备
        dry_run=base_args.dry_run,  # 是否 dry run
        num_samples=num_samples,  # 当前分块采样数
        seed=seed,  # 当前分块种子
        search_space=base_args.search_space,  # 使用的搜索空间
        top_k_percent=top_k_percent,  # Top-K 截断比例（仅 myscore 需要）
        alpha_threshold=alpha_threshold,  # 关键层筛选阈值（仅 myscore 需要）
    )

# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
def run_full_experiment(args):
    """执行完整批量实验"""  # 函数注释
    # 加载当前指定的 Proxy 评估函数
    evaluate_task_func, proxy_module = load_proxy_module(args.proxy)  # 动态加载
    compute_scores = proxy_module.compute_scores  # 获取该模块的相关性计算函数

    log_root = ensure_dir(Path(args.log_root))  # 确保日志根目录存在
    
    # 实验名增加 Proxy 和 Search Space 标识，避免覆盖
    base_exp_name = args.exp_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_sub_name = f"{base_exp_name}_{args.proxy}_{args.search_space}"
    exp_dir = ensure_dir(log_root / exp_sub_name)  # 创建实验目录

    config = {  # 记录实验配置
        "proxy": args.proxy,  # 当前使用的 Proxy 方法
        "tasks": args.tasks,  # 任务列表
        "search_space": args.search_space,  # 搜索空间（当前这一次运行所用的空间）
        "total_samples": args.total_samples,  # 总评估数量
        "chunk_size": args.chunk_size,  # 分块大小
        "batch_size": args.batch_size,  # batch 大小
        "maxbatch": args.maxbatch,  # 截断 batch 数
        "seed": args.seed,  # 基础随机种子
        "device": args.device,  # 运行设备
        "data_root": args.data_root,  # 数据目录
        "dry_run": args.dry_run,  # 是否 dry run
        "created_at": datetime.now().isoformat(),  # 创建时间
    }
    save_json(config, exp_dir / "config.json")  # 保存配置文件

    global_seed = args.seed  # 基础随机种子
    total_chunks = (args.total_samples + args.chunk_size - 1) // args.chunk_size  # 计算总分块数（向上取整）

    # === 关键修改：按任务分组，而不是按 chunk 分组 ===
    # 先完成所有 chunk 的 task1，再完成所有 chunk 的 task2，最后完成所有 chunk 的 task3
    for task in args.tasks:  # 外层循环：遍历每个任务
        print(f"\n===== 开始评估任务: {task} =====")  # 打印任务开始
        
        remaining = args.total_samples  # 重置剩余待评估数量
        chunk_id = 0  # 重置分块计数器

        # 使用 tqdm 显示 chunk 进度条
        pbar = tqdm(total=total_chunks, desc=f"[{args.proxy}][{args.search_space}][{task}] Chunks", unit="chunk")
        
        while remaining > 0:  # 逐块执行当前任务
            chunk_id += 1  # 当前分块编号
            num_samples = min(args.chunk_size, remaining)  # 当前分块采样数
            chunk_seed = global_seed + chunk_id  # 为分块生成新种子

            chunk_args = make_args_namespace(args, num_samples=num_samples, seed=chunk_seed)  # 生成分块参数
            
            task_start = time.time()  # 记录任务开始时间
            res = evaluate_task_func(task, args.search_space, chunk_args)  # 调用对应的 Proxy 评估函数
            task_elapsed = time.time() - task_start  # 计算任务耗时
            
            # 为每个任务单独构造分块日志
            task_chunk_log = {  # 构造任务分块日志
                "chunk_id": chunk_id,  # 分块编号
                "task": task,  # 任务名称
                "num_samples": num_samples,  # 本块采样数
                "seed": chunk_seed,  # 本块随机种子
                "elapsed_sec": task_elapsed,  # 本任务耗时
                "result": res,  # 本任务结果（单个任务的结果）
            }
            
            # 为每个任务单独保存 chunk 文件
            task_chunk_path = exp_dir / f"chunk_{chunk_id:03d}_{task}.json"  # 任务分块日志路径（文件名包含任务名）
            save_json(task_chunk_log, task_chunk_path)  # 写出任务分块日志
            
            # === 关键修复：立即释放内存引用，避免累积 ===
            del res  # 删除 res 引用，释放可能持有的 Tensor / DataLoader 引用
            del task_chunk_log  # 删除 log 引用（已保存到文件）
            
            # 每个 chunk 结束后立即清理
            if torch.cuda.is_available():  # 如果有 GPU
                torch.cuda.empty_cache()  # 清理 CUDA 缓存
            gc.collect()  # 强制垃圾回收

            remaining -= num_samples  # 更新剩余数量
            pbar.update(1)  # 更新进度条
            pbar.set_postfix({"samples": num_samples, "time": f"{task_elapsed:.1f}s"})  # 显示额外信息
        
        pbar.close()  # 关闭进度条
        
        # === 当前任务的所有 chunk 完成后，做一次彻底清理 ===
        print(f"===== 任务 {task} 全部完成，开始清理... =====")
        if torch.cuda.is_available():  # 如果有 GPU 设备
            torch.cuda.empty_cache()  # 主动清理 CUDA 缓存，减少显存碎片
        gc.collect()  # 强制 Python 垃圾回收，释放临时对象引用

    # --------------------- 为每个任务单独聚合与总结输出 ---------------------
    # 关键修复：从文件重新读取，而不是从内存中的 task_chunks 读取
    for task in args.tasks:  # 遍历每个任务
        all_gt_scores = []  # 当前任务的 GT 列表
        all_proxy_scores = []  # 当前任务的 Proxy 分数列表
        total_elapsed = 0.0  # 当前任务总耗时累积
        chunk_file_list = []  # 当前任务的分块文件列表

        # 从文件系统重新读取所有 chunk（避免内存累积）
        for cid in range(1, total_chunks + 1):  # 遍历所有分块 ID
            chunk_file = exp_dir / f"chunk_{cid:03d}_{task}.json"  # 构造文件路径
            if not chunk_file.exists():  # 如果文件不存在（理论上不应该发生）
                continue  # 跳过
            
            with chunk_file.open("r", encoding="utf-8") as f:  # 打开文件
                task_chunk = json.load(f)  # 读取 JSON
            
            total_elapsed += task_chunk["elapsed_sec"]  # 累加耗时
            task_res = task_chunk["result"]  # 获取任务结果
            all_gt_scores.extend(task_res["gt"])  # 汇总 GT
            all_proxy_scores.extend(task_res["pred"])  # 汇总 Proxy 分数
            chunk_file_list.append(f"chunk_{cid:03d}_{task}.json")  # 记录分块文件名

        if all_gt_scores and not args.dry_run:  # 若有数据且不是 dry run
            final_corr = compute_scores(ytest=all_gt_scores, test_pred=all_proxy_scores)  # 计算当前任务的最终相关性
        else:
            final_corr = {"kendalltau": None, "spearman": None}  # 无法计算则返回空值

        kt = final_corr.get("kendalltau")  # 取 KendallTau
        sp = final_corr.get("spearman")  # 取 Spearman
        kt_str = f"{kt:.4f}" if isinstance(kt, (int, float)) else "N/A"  # 格式化显示
        sp_str = f"{sp:.4f}" if isinstance(sp, (int, float)) else "N/A"  # 格式化显示

        task_summary = {  # 构造当前任务的汇总信息
            "config": config,  # 记录配置
            "task": task,  # 任务名称
            "num_chunks": total_chunks,  # 分块数量（使用实际分块数）
            "chunks": chunk_file_list,  # 分块文件列表
            "total_samples_evaluated": len(all_gt_scores),  # 聚合的样本数量
            "total_elapsed_sec": total_elapsed,  # 总耗时
            "final_aggregated_corr": final_corr,  # 聚合相关性
        }
        task_summary_path = exp_dir / f"summary_{task}.json"  # 任务 summary 路径（文件名包含任务名）
        save_json(task_summary, task_summary_path)  # 写入任务 summary

        print(f"\n[SUMMARY] [{args.proxy}] [{args.search_space}] [{task}] 聚合 {len(all_gt_scores)} 个样本的最终相关性：")  # 打印标题
        print(f"KendallTau: {kt_str}, Spearman: {sp_str}")  # 打印相关性
        print(f"总耗时：{total_elapsed:.1f}s")  # 打印耗时

    print(f"=== 全部完成，日志位于 {exp_dir} ===")  # 完成提示

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    """解析命令行参数"""  # 函数注释
    parser = argparse.ArgumentParser(description="Full runner for TransNAS Proxy experiments")  # 创建解析器
    parser.add_argument("--tasks", nargs="+", default=["autoencoder", "segmentsemantic", "normal"], help="任务列表")  # 任务参数
    
    # Proxy 选择：支持单选或多选列表
    parser.add_argument("--proxies", nargs="+", choices=["zico", "myscore"], default=None, help="要顺序运行的 Proxy 列表，例如: zico myscore")
    parser.add_argument("--proxy", choices=["zico", "myscore"], default=None, help="单个 Proxy（兼容旧用法，建议使用 --proxies）")
    
    # Search Space 选择：支持单选或多选列表
    parser.add_argument("--search_spaces", nargs="+", choices=["micro", "macro"], default=None, help="要顺序运行的搜索空间列表，例如: micro macro")
    parser.add_argument("--search_space", choices=["micro", "macro"], default=None, help="单个搜索空间（兼容旧用法，建议使用 --search_spaces）")
    
    parser.add_argument("--total_samples", type=int, default=200, help="总共要评估多少个架构（分块执行）")  # 总采样数
    parser.add_argument("--chunk_size", type=int, default=20, help="每个分块一次评估多少个架构")  # 分块大小
    parser.add_argument("--batch_size", type=int, default=128, help="DataLoader 的 batch 大小")  # batch 大小
    parser.add_argument("--maxbatch", type=int, default=2, help="截断的 batch 数")  # 截断 batch 数
    parser.add_argument("--seed", type=int, default=42, help="基础随机种子")  # 随机种子
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="运行设备")  # 设备
    parser.add_argument("--dry_run", action="store_true", help="仅跑数据管线，不计算 Proxy")  # dry run 开关
    parser.add_argument("--data_root", type=str, default=str(CURRENT_DIR.parent / "NASLib" / "data"), help="数据根目录")  # 数据路径
    parser.add_argument("--log_root", type=str, default=str(CURRENT_DIR / "full_run_logs"), help="日志根目录（自动创建）")  # 日志目录
    parser.add_argument("--exp_name", type=str, default=None, help="自定义实验子目录名称，不填则用时间戳")  # 实验名
    return parser.parse_args()  # 返回解析结果


if __name__ == "__main__":  # 入口判断
    args = parse_args()  # 解析参数

    if args.total_samples <= 0:  # 检查总采样数
        raise ValueError("total_samples 必须 > 0")  # 抛出错误
    if args.chunk_size <= 0:  # 检查分块大小
        raise ValueError("chunk_size 必须 > 0")  # 抛出错误

    # 解析需要顺序运行的 Search Space 列表
    if args.search_spaces is not None:
        search_spaces = args.search_spaces
    elif args.search_space is not None:
        search_spaces = [args.search_space]
    else:
        search_spaces = ["micro"]  # 默认

    # 解析需要顺序运行的 Proxy 列表
    if args.proxies is not None:
        proxies = args.proxies
    elif args.proxy is not None:
        proxies = [args.proxy]
    else:
        proxies = ["zico"]  # 默认只跑 zico，兼容旧行为

    # 双重循环：依次运行每个 Proxy 和每个搜索空间
    for proxy in proxies:
        for ss in search_spaces:
            # 构造子参数对象
            sub_args_dict = vars(args).copy()
            sub_args_dict["proxy"] = proxy  # 设置当前 Proxy
            sub_args_dict["search_space"] = ss  # 设置当前 Search Space
            sub_args = argparse.Namespace(**sub_args_dict)

            print(f"\n===== 开始运行: Proxy={proxy}, Space={ss} =====")  # 打印信息
            try:
                run_full_experiment(sub_args)  # 运行主流程
            except Exception as e:
                print(f"!!! 运行出错 Proxy={proxy}, Space={ss}: {e}")
                import traceback
                traceback.print_exc()
