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
import itertools  # 截断数据迭代
import importlib.util  # 动态加载模块

import torch  # 张量与设备管理
import numpy as np  # 数值计算
import warnings  # 警告控制
from tqdm import tqdm  # 进度条显示
from scipy.stats import rankdata  # 排名计算
from fvcore.nn import FlopCountAnalysis  # FLOPs 计算

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


def select_architectures_by_percentile(ss, dataset_api, task: str, start_percent: float, end_percent: float, search_space_name: str):
    """
    根据 Ground Truth 百分比范围选择架构。
    
    参数：
        ss: 搜索空间对象
        dataset_api: TransNAS-Bench API
        task: 任务名称
        start_percent: 起始百分比（0-100），从性能最好开始计
        end_percent: 结束百分比（0-100）
        search_space_name: 搜索空间名称 ('micro' 或 'macro')，直接从命令行参数传入
    
    返回：
        arch_list: 架构字符串列表
    
    示例：
        - start_percent=0, end_percent=10  → 前 10% 的架构（性能最好）
        - start_percent=30, end_percent=50 → 30%-50% 的架构（中等性能）
        - start_percent=90, end_percent=100 → 后 10% 的架构（性能最差）
    """
    api = dataset_api['api']
    # 直接使用传入的 search_space_name，不再通过属性判断，避免误判
    
    print(f"正在查询所有 {search_space_name} 架构的 GT 分数...", end="", flush=True)
    
    # 获取所有架构
    all_archs = api.all_arch_dict[search_space_name]
    
    # 确定正确的 metric
    if task in ["autoencoder", "normal"]:
        metric_name = "valid_ssim"
    elif task == "segmentsemantic":
        metric_name = "valid_acc"
    elif task == "room_layout":
        metric_name = "valid_neg_loss"
    else:
        metric_name = "valid_top1"
    
    # 收集所有架构的 GT 分数
    arch_scores = []
    for arch_str in all_archs:
        try:
            gt_score = api.get_single_metric(arch_str, task, metric_name, mode="final")
            arch_scores.append((arch_str, gt_score))
        except:
            continue
    
    total_archs = len(arch_scores)
    print(f" 完成（共 {total_archs} 个有效架构）")
    
    # 排序（降序：分数越高越靠前，即性能最好的排在前面）
    arch_scores_sorted = sorted(arch_scores, key=lambda x: x[1], reverse=True)
    
    # 计算百分比对应的索引
    start_idx = int(total_archs * start_percent / 100)
    end_idx = int(total_archs * end_percent / 100)
    
    # 确保索引在有效范围内
    start_idx = max(0, min(start_idx, total_archs - 1))
    end_idx = max(start_idx + 1, min(end_idx, total_archs))
    


    # 选择架构
    selected_archs = arch_scores_sorted[start_idx:end_idx]
    num_selected = len(selected_archs)
    
    if num_selected > 0:
        print(f"✓ 选择 {start_percent:.1f}%-{end_percent:.1f}% 区间的 {num_selected} 个架构")
        print(f"  GT 范围: {selected_archs[-1][1]:.4f} (最低) ~ {selected_archs[0][1]:.4f} (最高)")
    else:
        print(f"警告：未选择到任何架构！")
    
    # 直接返回架构字符串列表（保持与 TransNASBenchAPI 的键一致）
    # 后续用到模型时再通过 set_spec 解析，避免二次编码带来的字符串不一致问题
    arch_list = [arch_str for arch_str, _ in selected_archs]
    return arch_list


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
# FLOPs 计算逻辑
# ============================================================================

def compute_flops(model, train_batches, device):
    """
    计算模型的 FLOPs（浮点运算次数）。
    
    注意：FLOPs 越低越好（效率更高），但在排名聚合中需要取反。
    """
    model.eval()  # 评估模式
    model.to(device)
    
    data, _ = train_batches[0]  # 取第一个 batch
    input_tensor = data[:1].to(device)  # 只用一个样本即可
    
    try:
        with torch.no_grad():
            # 抑制 fvcore 的 unsupported operator 警告
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*unsupported.*")
                warnings.filterwarnings("ignore", message=".*Unsupported.*")
                warnings.filterwarnings("ignore", message=".*were never called.*")
                flop_analyzer = FlopCountAnalysis(model, input_tensor)
                flop_analyzer.unsupported_ops_warnings(False)  # 关闭警告
                flop_analyzer.uncalled_modules_warnings(False)  # 关闭未调用模块警告
                total_flops = flop_analyzer.total()
        return float(total_flops)
    except Exception as e:
        print(f"!!! FLOPs 计算失败: {e}")
        return 0.0


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
    
    注意：
        - FLOPs 在传入前已取负数，因此"越低越好"已转换为"越高越好"
        - 所有 proxy 都是分数越高排名越高
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
        
        # 对于 micro 搜索空间，需要手动解析架构字符串
        if ss_name == "micro":
            parts = arch_str.split("-")
            config_part = parts[-1]
            config_parts = config_part.split("_")
            
            op_indices = []
            op_indices.append(int(config_parts[0]))
            for digit in config_parts[1]:
                op_indices.append(int(digit))
            for digit in config_parts[2]:
                op_indices.append(int(digit))
            
            graph.set_op_indices(op_indices)
        else:
            # macro 搜索空间可以直接使用 set_spec
            graph.set_spec(arch_str, dataset_api=dataset_api)
        
        if args.dry_run:
            arch_strs.append(arch_str)
            # dry_run 模式下也查一次 GT，保持接口一致
            # 根据任务选择正确的 metric 名称
            if task in ["autoencoder", "normal"]:
                metric_name = "valid_ssim"
            elif task == "segmentsemantic":
                metric_name = "valid_acc"
            elif task == "room_layout":
                metric_name = "valid_neg_loss"
            else:
                metric_name = "valid_top1"
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
                if proxy == "zico":
                    model.train()
                    score = compute_zico_score(model, train_batches, loss_fn, args.device)
                elif proxy == "naswot":
                    model.eval()
                    score = compute_naswot_score(model, train_batches, args.device)
                elif proxy == "flops":
                    model.eval()
                    score = compute_flops(model, train_batches, args.device)
                    # print(f"FLOPs: {score}")
                    # score = np.log(score)
                else:
                    score = 0.0
                
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
        if task in ["autoencoder", "normal"]:
            metric_name = "valid_ssim"
        elif task == "segmentsemantic":
            metric_name = "valid_acc"
        elif task == "room_layout":
            metric_name = "valid_neg_loss"
        else:
            metric_name = "valid_top1"
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
        aggregated_scores = [0.0] * len(arch_strs)
    
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
            ss, dataset_api, task, args.start_percent, args.end_percent, args.search_space
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
                print(f"  {proxy_name:12} only: τ={corr['kendalltau']:.4f}, ρ={corr['spearman']:.4f}")
        
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
                print(f"  {proxy_name:12} only: τ={corr['kendalltau']:.4f}, ρ={corr['spearman']:.4f}")
                best_single_tau = max(best_single_tau, corr['kendalltau'])
        
        # 打印聚合结果
        print(f"  {proxy_names:12}     : τ={res['kendalltau']:.4f}, ρ={res['spearman']:.4f}")
        
        # 打印改进程度
        improvement = res['kendalltau'] - best_single_tau
        print(f"  改进: τ {improvement:+.4f}")


if __name__ == "__main__":
    main()

