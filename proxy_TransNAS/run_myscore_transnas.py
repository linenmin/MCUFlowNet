################################################################################
# FILE: run_myscore_transnas.py
# PURPOSE: Evaluate the correlation between C-SWAG Proxy and Ground Truth 
#          performance on TransNAS-Bench-101.
#          
# MODE: This script operates in a **Sampling and Correlation Printing** mode. 
#       It randomly samples a small number of architectures, computes their C-SWAG 
#       scores, retrieves their GT scores, and prints the resulting Kendall Tau 
#       and Spearman correlation coefficients.
# 
# AUTHOR: Your Name / (fill in your affiliation)
# CREATION DATE: November 30, 2025
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
import numpy as np # 确保导入 numpy
import warnings # 警告控制

# 忽略特定的 FutureWarning，保持输出整洁
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*squared.*")

# 插入 ZiCo 与 NASLib 的代码路径
CURRENT_DIR = Path(__file__).resolve().parent  # 本文件所在目录
ROOT_DIR = CURRENT_DIR.parent  # MCUFlowNet 根目录
ZICO_ROOT = ROOT_DIR / "Zico_repo"  # ZiCo 代码根目录
NASLIB_ROOT = ROOT_DIR / "NASLib"  # NASLib 代码根目录
sys.path.insert(0, str(ZICO_ROOT))  # 优先导入 ZiCo
sys.path.insert(0, str(NASLIB_ROOT))  # 再导入 NASLib

# === 关键修改：导入自定义的 C-SWAG Proxy ===
from compute_myscore import compute_myscore_score as compute_proxy_score

from naslib import utils  # 统一从 utils 取函数
# 必须导入 ops，因为我们会用到 GenerativeDecoder
from naslib.search_spaces.core import primitives as ops

# 函数别名，便于调用
get_train_val_loaders = utils.get_train_val_loaders  # 数据加载
compute_scores = utils.compute_scores  # 相关性计算


def load_transbench_classes():
    """动态加载 transbench101 的搜索空间，避免触发顶层 __init__ 依赖。"""
    graph_path = NASLIB_ROOT / "naslib" / "search_spaces" / "transbench101" / "graph.py"  # 路径
    spec = importlib.util.spec_from_file_location("transbench_graph", graph_path)  # 创建规范
    module = importlib.util.module_from_spec(spec)  # 创建模块
    spec.loader.exec_module(module)  # 执行模块
    
    return module.TransBench101SearchSpaceMicro, module.TransBench101SearchSpaceMacro, module  # 返回类与模块


def load_transbench_api(data_root: Path, task: str):
    """直接加载 TransNASBenchAPI，避免 naslib.search_spaces 顶层依赖。"""
    api_path = NASLIB_ROOT / "naslib" / "search_spaces" / "transbench101" / "api.py"  # API 路径
    spec = importlib.util.spec_from_file_location("transbench_api", api_path)  # 创建规范
    module = importlib.util.module_from_spec(spec)  # 创建模块
    spec.loader.exec_module(module)  # 执行模块
    TransNASBenchAPI = module.TransNASBenchAPI  # 取类
    # 默认查找 data_root 下的 pth，否则回落 NASLib 自带路径
    candidate = data_root / "transnas-bench_v10141024.pth"  # 用户指定路径
    if not candidate.exists():
        candidate = NASLIB_ROOT / "naslib" / "data" / "transnas-bench_v10141024.pth"  # 回落默认
    assert candidate.exists(), f"缺少 transnas-bench_v10141024.pth，检查 {candidate}"  # 断言存在
    api = TransNASBenchAPI(str(candidate))  # 实例化
    return {"api": api, "task": task}  # 返回与 NASLib 一致的字典


def build_config(data_root: Path, dataset: str, batch_size: int, seed: int):
    """构造最小配置对象，复用 NASLib 的 get_train_val_loaders。"""
    # 用简单对象而非 CfgNode，避免修改库代码
    search_cfg = argparse.Namespace(
        seed=seed,  # 随机种子
        batch_size=batch_size,  # batch 大小
        train_portion=0.7,  # 训练集比例
    )
    config = argparse.Namespace(
        data=str(data_root),  # 数据根路径
        dataset=dataset,  # 任务名
        search=search_cfg,  # 训练相关配置
    )
    return config  # 返回配置


def make_train_loader(task: str, data_root: Path, batch_size: int, seed: int):
    """生成指定任务的训练 DataLoader，只取训练队列。"""
    config = build_config(data_root, task, batch_size, seed)  # 构建配置
    train_loader, _, _, _, _ = get_train_val_loaders(config)  # 取训练 loader
    return train_loader  # 返回 loader


def truncate_loader(loader, max_batch: int):
    """截断 DataLoader 只保留前 max_batch 个 batch。"""
    # 转为列表以兼容 compute_zico 里的 enumerate
    return list(itertools.islice(iter(loader), max_batch))  # 返回前若干 batch


class SegmentationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, logits, label):
        # 1. 确保 Label 是 Long 类型
        if label.dtype != torch.long:
            label = label.long()

        # 2. 处理 Label 维度
        # 如果是 (B, 1, H, W)，squeeze 成 (B, H, W)
        if label.ndim == 4 and label.shape[1] == 1:
            label = label.squeeze(1)
            
        # 3. 检查 Logits 维度并适配
        # 语义分割标准：Logits (B, C, H, W), Label (B, H, W)
        if logits.ndim == 4 and label.ndim == 3:
            return self.ce(logits, label)
            
        # 4. 如果以上都不对，尝试 Flatten 策略 (兼容性最强)
        # 将 Logits 转为 (N, C)，Label 转为 (N)
        if logits.ndim > 2:
            num_classes = logits.shape[1]
            # permute 把 channel 放到最后: (B, H, W, C)
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, num_classes)
            label_flat = label.reshape(-1)
            return self.ce(logits_flat, label_flat)
            
        # 如果已经是 2D Input 和 1D Target，直接计算
        return self.ce(logits, label)

def get_loss_fn(task: str):
    """根据任务选择合适的损失函数，遵循 NASLib 的定义。"""
    # autoencoder/normal 用 L1，segmentsemantic 用交叉熵
    if task in ["autoencoder", "normal"]:
        return torch.nn.L1Loss()
    elif task == "segmentsemantic":
        # 使用自定义 Loss 包装器来处理维度和类型
        return SegmentationLoss()
    else:
        # 其他分类任务可扩展，此处默认交叉熵
        return torch.nn.CrossEntropyLoss()

# 
# 
# 
def sample_architectures(ss, dataset_api, num_samples: int, seed: int):
    """随机采样若干架构并解析成可前向的模型。"""
    random.seed(seed)  # 控制随机
    torch.manual_seed(seed)  # 控制随机
    samples = []  # 存储图与哈希
    for _ in range(num_samples):
        graph = ss.clone()  # 复制搜索空间
        graph.sample_random_architecture(dataset_api=dataset_api)  # 随机采样架构
        graph.parse()  # 实例化模型
        samples.append(graph)  # 收集
    return samples  # 返回列表


def evaluate_task(task: str, ss_name: str, args):
    """在单个任务上评估 C-SWAG 与真值的相关性。"""
    # 动态取 Metric 与搜索空间类
    TransBench101SearchSpaceMicro, TransBench101SearchSpaceMacro, graph_module = load_transbench_classes()  # 取类和模块
    Metric = graph_module.Metric  # 直接使用模块中的 Metric，避免类型不一致
    # 数据与 API 准备
    data_root = Path(args.data_root).resolve()  # 数据路径
    dataset_api = load_transbench_api(data_root, task)  # 加载表格 API
    if args.dry_run:
        train_batches = []  # dry run 不需要数据
        loss_fn = None  # 占位
    else:
        train_loader = make_train_loader(task, data_root, args.batch_size, args.seed)  # 构造训练 loader
        train_batches = truncate_loader(train_loader, args.maxbatch)  # 截断 batch
        loss_fn = get_loss_fn(task).to(args.device)  # 选择损失并放到设备

    # 搜索空间与架构采样
    # 创建搜索空间（直接使用已加载的类，避免重复加载）
    if ss_name == "micro":
        # 显式指定 n_classes 为 17 (segmentsemantic 的类别数)
        if task == "segmentsemantic":
             ss = TransBench101SearchSpaceMicro(dataset=task, create_graph=True, n_classes=17)
        else:
             ss = TransBench101SearchSpaceMicro(dataset=task, create_graph=True)
    else:
        # 错误修正：必须传入 dataset=task，否则默认为 jigsaw
        ss = TransBench101SearchSpaceMacro(dataset=task, create_graph=True)
    
    samples = sample_architectures(ss, dataset_api, args.num_samples, args.seed)  # 采样架构

    gt_scores = []  # 存储真值
    proxy_scores = []  # 存储 C-SWAG
    arch_hashes = []  # 存储每个架构的哈希（op_indices）
    start = time.time()  # 计时开始
    
    for i, graph in enumerate(samples):
        # 记录当前架构的哈希（使用 op_indices 作为离散结构编码，便于后续分析）
        try:
            arch_hash = list(graph.get_hash())
        except Exception:
            # 如果意外失败，则用 None 占位，避免中断主流程
            arch_hash = None
        arch_hashes.append(arch_hash)
        
        # 查询真值（VAL_ACCURACY 映射到任务对应指标）
        gt = graph.query(metric=Metric.VAL_ACCURACY, dataset=task, dataset_api=dataset_api)  # 取真值
        gt_scores.append(gt)  # 记录真值
        
        if args.dry_run:
            proxy_scores.append(0.0)  # 仅占位
            continue  # 跳过计算
            
        # === 关键 OOM 修复 ===
        model = graph.to(args.device)  # 取具体模型并放设备
        model.train()  # 训练模式
        
        # === 关键修改：调用 C-SWAG Proxy ===
        try:
            zc = compute_proxy_score(model, train_batches, loss_fn, args.device)  # 计算 Proxy 分数
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"!!! OOM detected at sample {i}. Trying to clear cache and retry...")
                torch.cuda.empty_cache()
                zc = 0.0
            else:
                raise e
        
        proxy_scores.append(zc)  # 记录分数
        
        # === 显存清理 ===
        model.cpu() # 关键：必须移回 CPU
        del model  # 释放模型
        torch.cuda.empty_cache()  # 清理显存

        # === 新增：强制 GC ===
        import gc
        gc.collect()

    elapsed = time.time() - start  # 总耗时
    # 相关性评估
    corr = compute_scores(ytest=gt_scores, test_pred=proxy_scores)  # 计算相关性
    return {
        "task": task,  # 任务名
        "kendalltau": corr.get("kendalltau"),  # Kendall Tau
        "spearman": corr.get("spearman"),  # Spearman
        "time": elapsed,  # 用时
        "gt": gt_scores,  # 真值列表（与架构顺序一致）
        "pred": proxy_scores,  # 预测列表（与架构顺序一致）
        "arch_hash": arch_hashes,  # 架构哈希列表（每个元素通常是长度为 6 的 op_indices 列表）
    }  # 返回结果字典

#python run_myscore_transnas.py --tasks autoencoder --search_space micro --num_samples 5 --batch_size 8
def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run C-SWAG on TransNAS-Bench-101")  # 创建解析器
    parser.add_argument("--tasks", nargs="+", default=["autoencoder", "segmentsemantic", "normal"], help="任务列表")  # 任务参数
    parser.add_argument("--search_space", choices=["micro", "macro"], default="micro", help="搜索空间类型")  # 空间选择
    parser.add_argument("--data_root", type=str, default=str(NASLIB_ROOT / "data"), help="数据根路径")  # 数据路径
    parser.add_argument("--num_samples", type=int, default=20, help="采样架构数量")  # 采样数量
    parser.add_argument("--batch_size", type=int, default=128, help="DataLoader 的 batch 大小")  # batch 大小
    parser.add_argument("--maxbatch", type=int, default=2, help="截断的 batch 数")  # 截断 batch 数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")  # 种子
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")  # 设备选择
    parser.add_argument("--dry_run", action="store_true", help="仅跑数据管线，不计算 Proxy")  # dry run
    return parser.parse_args()  # 返回参数


def main():
    """主入口，遍历任务并打印结果。"""
    args = parse_args()  # 解析参数
    torch.manual_seed(args.seed)  # 设定种子
    random.seed(args.seed)  # 设定种子
    results = []  # 结果列表
    for task in args.tasks:
        print(f"==> 评估任务 {task}")  # 打印任务
        res = evaluate_task(task, args.search_space, args)  # 评估单任务
        results.append(res)  # 收集结果
        # 打印简要结果
        print(
            f"[{task}] kendalltau={res['kendalltau']:.4f} spearman={res['spearman']:.4f} time={res['time']:.1f}s"
        )  # 打印相关性与时间

    # 汇总输出
    print("=== 汇总结果 ===")  # 标题
    for res in results:
        print(
            f"{res['task']}: kendalltau={res['kendalltau']:.4f}, spearman={res['spearman']:.4f}, time={res['time']:.1f}s"
        )  # 每任务结果


if __name__ == "__main__":
    main()  # 运行主函数

