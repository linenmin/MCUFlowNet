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
sys.path.insert(0, str(NASLIB_ROOT))  # 导入 NASLib

# === 全局缓存 TransNASBenchAPI，避免重复从磁盘加载巨大 .pth 文件 ===
_GLOBAL_TRANSNASBENCH_API = None  # 全局缓存变量，初始为 None

from naslib import utils  # 统一从 utils 取函数

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
    """直接加载 TransNASBenchAPI，并增加全局缓存，防止 full_run 循环中重复加载导致 RAM 爆掉。"""
    global _GLOBAL_TRANSNASBENCH_API  # 引用全局变量

    # 1. 动态加载模块定义（只负责拿到类定义，开销很小）
    api_path = NASLIB_ROOT / "naslib" / "search_spaces" / "transbench101" / "api.py"  # API 路径
    spec = importlib.util.spec_from_file_location("transbench_api", api_path)  # 创建规范
    module = importlib.util.module_from_spec(spec)  # 创建模块
    spec.loader.exec_module(module)  # 执行模块
    TransNASBenchAPI = module.TransNASBenchAPI  # 取类

    # 2. 如果全局缓存还没有实例，则从磁盘加载一次巨大的 .pth
    if _GLOBAL_TRANSNASBENCH_API is None:  # 只有第一次会进来
        # 默认查找 data_root 下的 pth，否则回落 NASLib 自带路径
        candidate = data_root / "transnas-bench_v10141024.pth"  # 用户指定路径
        if not candidate.exists():  # 如果用户路径不存在
            candidate = NASLIB_ROOT / "naslib" / "data" / "transnas-bench_v10141024.pth"  # 回落默认
        assert candidate.exists(), f"缺少 transnas-bench_v10141024.pth，检查 {candidate}"  # 断言存在
        # 实例化并缓存在全局变量中
        _GLOBAL_TRANSNASBENCH_API = TransNASBenchAPI(str(candidate))  # 只在第一次从磁盘读取

    # 3. 返回缓存的 API 对象（后续调用全部走内存，不再读盘）
    return {"api": _GLOBAL_TRANSNASBENCH_API, "task": task}  # 返回与 NASLib 一致的字典


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
    return list(itertools.islice(iter(loader), max_batch))  # 返回前若干 batch


def sample_architecture_identifiers(ss, dataset_api, num_samples: int, seed: int):
    """
    随机采样若干不重复的架构，只返回轻量的架构标识符（op_indices）。
    
    优势：
    - 内存占用极小（每个标识符只有几十字节）
    - 自动去重，保证没有重复架构
    - 可通过 set_op_indices 重建完整架构
    """
    random.seed(seed)  # 控制随机
    torch.manual_seed(seed)  # 控制随机
    
    arch_identifiers = []  # 只存储轻量的架构标识符（tuple of op_indices）
    seen_hashes = set()    # 用于去重
    
    max_attempts = num_samples * 10  # 最大尝试次数，避免死循环
    attempts = 0
    
    print(f"正在采样 {num_samples} 个不重复的架构...", end="", flush=True)
    
    while len(arch_identifiers) < num_samples and attempts < max_attempts:
        attempts += 1
        
        # 临时创建 graph，获取哈希
        temp_graph = ss.clone()
        temp_graph.sample_random_architecture(dataset_api=dataset_api)
        
        try:
            arch_hash = tuple(temp_graph.get_hash())  # 转 tuple 才能加入 set
        except Exception:
            del temp_graph
            continue  # 如果获取哈希失败，跳过
        
        # 去重检查
        if arch_hash in seen_hashes:
            del temp_graph  # 重复了，丢弃
            continue
        
        seen_hashes.add(arch_hash)
        arch_identifiers.append(arch_hash)
        del temp_graph  # 立即释放，不保留
    
    if len(arch_identifiers) < num_samples:
        print(f"\n警告：只采样到 {len(arch_identifiers)} 个不重复架构（目标 {num_samples}），可能搜索空间太小")
    else:
        print(f" 完成（尝试 {attempts} 次）")
    
    return arch_identifiers  # 返回轻量标识符列表


# ============================================================================
# NASWOT 核心计算逻辑（改编自 compute_NASWOT_score.py）
# ============================================================================

def network_weight_gaussian_init(net: torch.nn.Module):
    """使用高斯分布初始化网络权重，BN 层初始化为 1 和 0。"""
    with torch.no_grad():  # 无梯度模式
        for m in net.modules():  # 遍历所有模块
            if isinstance(m, torch.nn.Conv2d):  # 卷积层
                torch.nn.init.normal_(m.weight)  # 高斯初始化权重
                if hasattr(m, 'bias') and m.bias is not None:  # 如果有偏置
                    torch.nn.init.zeros_(m.bias)  # 偏置初始化为 0
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):  # BN 或 GN 层
                torch.nn.init.ones_(m.weight)  # 权重初始化为 1
                torch.nn.init.zeros_(m.bias)  # 偏置初始化为 0
            elif isinstance(m, torch.nn.Linear):  # 全连接层
                torch.nn.init.normal_(m.weight)  # 高斯初始化权重
                if hasattr(m, 'bias') and m.bias is not None:  # 如果有偏置
                    torch.nn.init.zeros_(m.bias)  # 偏置初始化为 0
            else:
                continue  # 其他层跳过
    return net  # 返回初始化后的网络


def logdet(K):
    """计算矩阵的对数行列式（log determinant）。"""
    s, ld = np.linalg.slogdet(K)  # 符号和对数行列式
    return ld  # 返回对数行列式


def compute_naswot_score(model, train_batches, device):
    """
    计算 NASWOT 分数（基于 ReLU 激活模式的核矩阵行列式）。
    
    参数：
        model: 待评估的神经网络模型
        train_batches: 真实数据批次列表（从 DataLoader 截断得到）
        device: 计算设备（cuda 或 cpu）
    
    返回：
        float: NASWOT 分数（核矩阵的对数行列式）
    """
    model = model.to(device)  # 模型移到设备
    network_weight_gaussian_init(model)  # 高斯初始化权重
    
    # 从真实数据中获取输入（使用第一个 batch）
    data, _ = train_batches[0]  # 取第一个 batch 的数据和标签
    input_tensor = data.to(device)  # 输入移到设备
    batch_size = input_tensor.size(0)  # 从数据获取 batch 大小
    
    model.K = np.zeros((batch_size, batch_size))  # 初始化核矩阵 K
    
    def counting_forward_hook(module, inp, out):
        """前向钩子：累积 ReLU 激活模式到核矩阵 K。"""
        try:
            if not module.visited_backwards:  # 如果还没被反向传播访问过，跳过
                return
            if isinstance(inp, tuple):  # 如果输入是 tuple，取第一个
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)  # 展平输入 (B, -1)
            x = (inp > 0).float()  # 二值化：激活为 1，未激活为 0
            K = x @ x.t()  # 计算激活模式的内积矩阵
            K2 = (1. - x) @ (1. - x.t())  # 计算未激活模式的内积矩阵
            model.K = model.K + K.cpu().numpy() + K2.cpu().numpy()  # 累加到核矩阵
        except Exception as err:
            print('---- NASWOT 计算错误：')
            print(model)
            raise err
    
    # 为所有 ReLU 层注册钩子
    for name, module in model.named_modules():  # 遍历所有模块
        # 适配 TransNAS：检查标准 PyTorch ReLU 层
        if isinstance(module, (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.PReLU)):
            module.visited_backwards = True  # 标记为已访问（让 forward_hook 生效）
            module.register_forward_hook(counting_forward_hook)  # 注册前向钩子
    
    # 只需前向传播即可触发 hooks 收集激活模式（无需反向传播）
    with torch.no_grad():  # 优化：不需要梯度，节省内存和计算
        _ = model(input_tensor)  # 前向传播触发 forward_hooks
    
    # 计算核矩阵的对数行列式作为分数
    score = logdet(model.K)  # 对数行列式
    
    return float(score)  # 返回分数


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

