################################################################################
# FILE: run_ZICO_NASWOT_transnas.py
# PURPOSE: Evaluate the correlation between aggregated ZiCo+NASWOT ensemble 
#          and Ground Truth performance on TransNAS-Bench-101.
#          
# MODE: Uses non-linear ranking aggregation from AZ-NAS paper to combine
#       multiple zero-shot proxies for better ranking consistency.
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
from scipy.stats import rankdata  # 排名计算

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
# ZiCo 核心计算逻辑（从 run_zico_transnas.py 复用）
# ============================================================================

class SegmentationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, logits, label):
        # 1. 确保 Label 是 Long 类型
        if label.dtype != torch.long:
            label = label.long()

        # 2. 处理 Label 维度
        if label.ndim == 4 and label.shape[1] == 1:
            label = label.squeeze(1)
            
        # 3. 检查 Logits 维度并适配
        if logits.ndim == 4 and label.ndim == 3:
            return self.ce(logits, label)
            
        # 4. Flatten 策略 (兼容性最强)
        if logits.ndim > 2:
            num_classes = logits.shape[1]
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, num_classes)
            label_flat = label.reshape(-1)
            return self.ce(logits_flat, label_flat)
            
        return self.ce(logits, label)


def get_loss_fn(task: str):
    """根据任务选择合适的损失函数。"""
    if task in ["autoencoder", "normal"]:
        return torch.nn.L1Loss()
    elif task == "segmentsemantic":
        return SegmentationLoss()
    else:
        return torch.nn.CrossEntropyLoss()


def getgrad_safe(model: torch.nn.Module, grad_dict: dict, step_iter=0):
    """安全获取梯度（处理 None 情况）。"""
    if step_iter == 0:
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Conv2d) or isinstance(mod, torch.nn.Linear):
                if mod.weight.grad is not None:
                    grad_dict[name] = [mod.weight.grad.data.cpu().reshape(-1).numpy()]
    else:
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Conv2d) or isinstance(mod, torch.nn.Linear):
                if mod.weight.grad is not None:
                    if name in grad_dict:
                        grad_dict[name].append(mod.weight.grad.data.cpu().reshape(-1).numpy())
    return grad_dict


def caculate_zico_safe(grad_dict):
    """安全计算 ZiCo 分数（处理空字典和零方差情况）。"""
    if not grad_dict:
        return 0.0
        
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname] = np.array(grad_dict[modname])
    
    nsr_mean_sum_abs = 0
    valid_layer_count = 0
    
    for j, modname in enumerate(grad_dict.keys()):
        nsr_std = np.std(grad_dict[modname], axis=0)
        if np.sum(nsr_std) == 0:
            continue
            
        nonzero_idx = np.nonzero(nsr_std)[0]
        if len(nonzero_idx) == 0:
            continue
            
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
        tmpsum = np.mean(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
        
        if tmpsum == 0:
            pass
        else:
            nsr_mean_sum_abs += np.log(tmpsum)
            valid_layer_count += 1

    if valid_layer_count > 0:
        return nsr_mean_sum_abs / valid_layer_count
    else:
        return 0.0


def compute_zico_score(model, train_batches, loss_fn, device: torch.device):
    """计算 ZiCo 分数。"""
    grad_dict = {}
    model.train()
    model.to(device)
    
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn.to(device)
    
    for i, batch in enumerate(train_batches):
        model.zero_grad()
        data, label = batch
        data = data.to(device)
        label = label.to(device)
        
        logits = model(data)
        loss = loss_fn(logits, label)
        loss.backward()
        
        grad_dict = getgrad_safe(model, grad_dict, i)
        model.zero_grad(set_to_none=True)
        
    res = caculate_zico_safe(grad_dict)
    del grad_dict
    
    return res


# ============================================================================
# NASWOT 核心计算逻辑（从 run_NASWOT_transnas.py 复用）
# ============================================================================

def network_weight_gaussian_init(net: torch.nn.Module):
    """使用高斯分布初始化网络权重。"""
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.Conv2d):
                if hasattr(m, 'weight') and m.weight is not None:
                    torch.nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    torch.nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                if hasattr(m, 'weight') and m.weight is not None:
                    torch.nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            else:
                continue
    return net


def logdet(K):
    """计算矩阵的对数行列式。"""
    s, ld = np.linalg.slogdet(K)
    return ld


def compute_naswot_score(model, train_batches, device):
    """计算 NASWOT 分数。"""
    model = model.to(device)
    network_weight_gaussian_init(model)
    
    data, _ = train_batches[0]
    input_tensor = data.to(device)
    batch_size = input_tensor.size(0)
    
    model.K = np.zeros((batch_size, batch_size))
    
    def counting_forward_hook(module, inp, out):
        try:
            if not module.visited_backwards:
                return
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            x = (inp > 0).float()
            K = x @ x.t()
            K2 = (1. - x) @ (1. - x.t())
            model.K = model.K + K.cpu().numpy() + K2.cpu().numpy()
        except Exception as err:
            print('---- NASWOT 计算错误：')
            print(model)
            raise err
    
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.PReLU)):
            module.visited_backwards = True
            module.register_forward_hook(counting_forward_hook)
    
    with torch.no_grad():
        _ = model(input_tensor)
    
    score = logdet(model.K)
    
    return float(score)


# ============================================================================
# 非线性排名聚合（AZ-NAS 方法）
# ============================================================================

def nonlinear_ranking_aggregation(proxy_scores_dict):
    """
    非线性排名聚合（来自 AZ-NAS 论文）。
    
    参数：
        proxy_scores_dict: 字典 {proxy_name: [score1, score2, ...]}
    
    返回：
        aggregated_scores: 聚合后的分数列表
    """
    m = len(next(iter(proxy_scores_dict.values())))  # 架构数量
    aggregated_scores = np.zeros(m)  # 初始化聚合分数
    
    print(f"\n=== 非线性排名聚合 ===")
    print(f"架构数量: {m}")
    print(f"Proxy 数量: {len(proxy_scores_dict)}")
    
    for proxy_name, scores in proxy_scores_dict.items():
        # 计算排名（升序：分数越高排名越大）
        ranks = rankdata(scores, method='ordinal')  # 排名从 1 到 m
        
        # 应用非线性聚合公式：log(rank / m)
        for i in range(m):
            aggregated_scores[i] += np.log(ranks[i] / m)  # 累加对数排名
        
        print(f"  {proxy_name}: 分数范围 [{np.min(scores):.4f}, {np.max(scores):.4f}]")
    
    print(f"聚合分数范围: [{np.min(aggregated_scores):.4f}, {np.max(aggregated_scores):.4f}]")
    
    return aggregated_scores.tolist()  # 返回列表


# ============================================================================
# 主评估逻辑
# ============================================================================

def evaluate_task(task: str, ss_name: str, args, shared_arch_identifiers=None):
    """在单个任务上评估 ZiCo+NASWOT 聚合与真值的相关性。"""
    # 动态取 Metric 与搜索空间类
    TransBench101SearchSpaceMicro, TransBench101SearchSpaceMacro, graph_module = load_transbench_classes()
    Metric = graph_module.Metric
    
    # 数据与 API 准备
    data_root = Path(args.data_root).resolve()
    dataset_api = load_transbench_api(data_root, task)
    
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
    
    # 如果提供了共享架构列表，直接使用；否则自己采样
    if shared_arch_identifiers is not None:
        arch_identifiers = shared_arch_identifiers
    else:
        arch_identifiers = sample_architecture_identifiers(ss, dataset_api, args.num_samples, args.seed)
    
    gt_scores = []  # 存储真值
    zico_scores = []  # 存储 ZiCo
    naswot_scores = []  # 存储 NASWOT
    arch_hashes = []  # 存储架构哈希
    start = time.time()
    
    # 逐个重建架构并评估
    for i, arch_identifier in enumerate(tqdm(arch_identifiers, desc=f"[{task}] 评估架构", unit="arch")):
        # 从标识符重建 graph
        graph = ss.clone()
        graph.set_op_indices(list(arch_identifier))
        
        if args.dry_run:
            arch_hashes.append(list(arch_identifier))
            gt = graph.query(metric=Metric.VAL_ACCURACY, dataset=task, dataset_api=dataset_api)
            gt_scores.append(gt)
            zico_scores.append(0.0)
            naswot_scores.append(0.0)
            del graph
            continue
        
        # === 实例化模型 ===
        graph.parse()
        model = graph.to(args.device)
        
        # === 计算 ZiCo 分数 ===
        try:
            model.train()  # ZiCo 需要训练模式
            zc = compute_zico_score(model, train_batches, loss_fn, args.device)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"!!! OOM (ZiCo) at sample {i}. Skipping...")
                torch.cuda.empty_cache()
                model.cpu()
                del model
                del graph
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                continue
            else:
                raise e
        except Exception as e:
            print(f"!!! Error (ZiCo) at sample {i}: {e}. Skipping...")
            model.cpu()
            del model
            del graph
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            continue
        
        # === 计算 NASWOT 分数 ===
        try:
            model.eval()  # NASWOT 需要评估模式
            nw = compute_naswot_score(model, train_batches, args.device)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"!!! OOM (NASWOT) at sample {i}. Skipping...")
                torch.cuda.empty_cache()
                model.cpu()
                del model
                del graph
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                continue
            else:
                raise e
        except Exception as e:
            print(f"!!! Error (NASWOT) at sample {i}: {e}. Skipping...")
            model.cpu()
            del model
            del graph
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            continue
        
        # 只有成功计算两个分数后，才记录数据
        arch_hashes.append(list(arch_identifier))
        gt = graph.query(metric=Metric.VAL_ACCURACY, dataset=task, dataset_api=dataset_api)
        gt_scores.append(gt)
        zico_scores.append(zc)
        naswot_scores.append(nw)
        
        # === 显存和内存清理 ===
        model.cpu()
        del model
        del graph
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    elapsed = time.time() - start
    
    # === 非线性排名聚合 ===
    if len(zico_scores) > 0 and not args.dry_run:
        proxy_scores = {
            "ZiCo": zico_scores,
            "NASWOT": naswot_scores,
        }
        aggregated_scores = nonlinear_ranking_aggregation(proxy_scores)
    else:
        aggregated_scores = [0.0] * len(arch_hashes)
    
    # 相关性评估
    corr = compute_scores(ytest=gt_scores, test_pred=aggregated_scores)
    
    # 同时计算单独 proxy 的相关性（用于对比）
    corr_zico = compute_scores(ytest=gt_scores, test_pred=zico_scores) if len(zico_scores) > 0 else {}
    corr_naswot = compute_scores(ytest=gt_scores, test_pred=naswot_scores) if len(naswot_scores) > 0 else {}
    
    return {
        "task": task,
        "kendalltau": corr.get("kendalltau"),
        "spearman": corr.get("spearman"),
        "time": elapsed,
        "gt": gt_scores,
        "pred": aggregated_scores,
        "zico": zico_scores,
        "naswot": naswot_scores,
        "arch_hash": arch_hashes,
        # 单独 proxy 的相关性（用于对比）
        "zico_kendalltau": corr_zico.get("kendalltau"),
        "zico_spearman": corr_zico.get("spearman"),
        "naswot_kendalltau": corr_naswot.get("kendalltau"),
        "naswot_spearman": corr_naswot.get("spearman"),
    }


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run ZiCo+NASWOT Ensemble on TransNAS-Bench-101")
    parser.add_argument("--tasks", nargs="+", default=["autoencoder", "segmentsemantic", "normal"], help="任务列表")
    parser.add_argument("--search_space", choices=["micro", "macro"], default="micro", help="搜索空间类型")
    parser.add_argument("--data_root", type=str, default=str(NASLIB_ROOT / "data"), help="数据根路径")
    parser.add_argument("--num_samples", type=int, default=20, help="采样架构数量")
    parser.add_argument("--batch_size", type=int, default=128, help="DataLoader 的 batch 大小")
    parser.add_argument("--maxbatch", type=int, default=2, help="截断的 batch 数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--dry_run", action="store_true", help="仅跑采样管线，不计算 proxy")
    return parser.parse_args()


def main():
    """主入口，遍历任务并打印结果。"""
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # === 在任务循环之前，先采样一次架构（所有任务共享）===
    print("=" * 80)
    print("Step 0: 为所有任务采样架构（共享）")
    print("=" * 80)
    
    # 创建搜索空间（用于采样）
    TransBench101SearchSpaceMicro, TransBench101SearchSpaceMacro, graph_module = load_transbench_classes()
    data_root = Path(args.data_root).resolve()
    dataset_api = load_transbench_api(data_root, args.tasks[0])
    
    # 创建搜索空间
    if args.search_space == "micro":
        if args.tasks[0] == "segmentsemantic":
            ss = TransBench101SearchSpaceMicro(dataset=args.tasks[0], create_graph=True, n_classes=17)
        else:
            ss = TransBench101SearchSpaceMicro(dataset=args.tasks[0], create_graph=True)
    else:
        ss = TransBench101SearchSpaceMacro(dataset=args.tasks[0], create_graph=True)
    
    shared_arch_identifiers = sample_architecture_identifiers(ss, dataset_api, args.num_samples, args.seed)
    print(f"✓ 采样完成，所有任务将共享这 {len(shared_arch_identifiers)} 个架构\n")
    
    # === 遍历任务，传入共享的架构列表 ===
    results = []
    for task in args.tasks:
        print(f"==> 评估任务 {task}")
        res = evaluate_task(task, args.search_space, args, shared_arch_identifiers)
        results.append(res)
        
        # 打印详细结果（包含单独 proxy 的对比）
        print(f"\n[{task}] 结果对比:")
        print(f"  ZiCo only:       kendalltau={res.get('zico_kendalltau', 0.0):.4f}, spearman={res.get('zico_spearman', 0.0):.4f}")
        print(f"  NASWOT only:     kendalltau={res.get('naswot_kendalltau', 0.0):.4f}, spearman={res.get('naswot_spearman', 0.0):.4f}")
        print(f"  ZiCo+NASWOT:     kendalltau={res['kendalltau']:.4f}, spearman={res['spearman']:.4f}")
        print(f"  用时: {res['time']:.1f}s\n")
    
    # 汇总输出
    print("=" * 80)
    print("=== 汇总结果 ===")
    print("=" * 80)
    for res in results:
        print(f"\n{res['task']}:")
        print(f"  ZiCo only:       τ={res.get('zico_kendalltau', 0.0):.4f}, ρ={res.get('zico_spearman', 0.0):.4f}")
        print(f"  NASWOT only:     τ={res.get('naswot_kendalltau', 0.0):.4f}, ρ={res.get('naswot_spearman', 0.0):.4f}")
        print(f"  ZiCo+NASWOT:     τ={res['kendalltau']:.4f}, ρ={res['spearman']:.4f}")
        print(f"  改进: τ +{res['kendalltau'] - max(res.get('zico_kendalltau', 0), res.get('naswot_kendalltau', 0)):.4f}")


if __name__ == "__main__":
    main()

