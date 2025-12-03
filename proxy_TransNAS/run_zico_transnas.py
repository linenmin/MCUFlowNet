################################################################################
# FILE: zico_correlation_sampler.py (Suggested Filename)
# PURPOSE: Evaluate the correlation between ZiCo (Zero-Shot Proxy) and Ground 
#          Truth performance on TransNAS-Bench-101.
#          
# MODE: This script operates in a **Sampling and Correlation Printing** mode. 
#       It randomly samples a small number of architectures, computes their ZiCo 
#       scores, retrieves their GT scores, and prints the resulting Kendall Tau 
#       and Spearman correlation coefficients.
# 
# AUTHOR: Enmin Lin/ KU Leuven
# CREATION DATE: November 26, 2025
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
from tqdm import tqdm  # 进度条显示

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

# === 新增：全局缓存 TransNASBenchAPI，避免重复从磁盘加载巨大 .pth 文件 ===
_GLOBAL_TRANSNASBENCH_API = None  # 全局缓存变量，初始为 None

# from ZeroShotProxy import compute_zico  # ZiCo 核心函数 (本文件使用本地优化的 compute_zico_score)
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
    """截断 DataLoader 只保留前 max_batch 个 batch（返回迭代器，节省内存）。"""
    # 返回迭代器而非列表，避免一次性加载所有数据到内存
    return itertools.islice(iter(loader), max_batch)  # 返回迭代器


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
    """随机采样若干架构，但不立即实例化模型（延迟到使用时）。"""
    random.seed(seed)  # 控制随机
    torch.manual_seed(seed)  # 控制随机
    samples = []  # 存储图（只保存架构描述，不实例化模型）
    for _ in range(num_samples):
        graph = ss.clone()  # 复制搜索空间
        graph.sample_random_architecture(dataset_api=dataset_api)  # 随机采样架构
        # 关键修改：不调用 graph.parse()，延迟实例化到真正需要时
        samples.append(graph)  # 收集架构描述
    return samples  # 返回架构列表（未实例化）


# === 新增：自定义的 ZiCo 计算逻辑，修复梯度为 None 的问题 ===
def getgrad_safe(model: torch.nn.Module, grad_dict: dict, step_iter=0):
    if step_iter == 0:
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Conv2d) or isinstance(mod, torch.nn.Linear):
                # 关键修复：检查 grad 是否为 None
                if mod.weight.grad is not None:
                    grad_dict[name] = [mod.weight.grad.data.cpu().reshape(-1).numpy()]
    else:
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Conv2d) or isinstance(mod, torch.nn.Linear):
                if mod.weight.grad is not None:
                    if name in grad_dict: # 确保 key 存在
                        grad_dict[name].append(mod.weight.grad.data.cpu().reshape(-1).numpy())
    return grad_dict

def caculate_zico_safe(grad_dict):
    allgrad_array = None
    # 如果 grad_dict 为空（比如所有层都没梯度），返回 0
    if not grad_dict:
        return 0.0
        
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname] = np.array(grad_dict[modname])
    
    nsr_mean_sum_abs = 0
    valid_layer_count = 0  # 新增：统计有效层数
    
    for j, modname in enumerate(grad_dict.keys()):
        nsr_std = np.std(grad_dict[modname], axis=0)
        if np.sum(nsr_std) == 0: # 避免全0导致的除0警告
            continue
            
        nonzero_idx = np.nonzero(nsr_std)[0]
        if len(nonzero_idx) == 0:
            continue
            
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
        # tmpsum = np.sum(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
        tmpsum = np.mean(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx]) #取平均而不是求和
        
        if tmpsum == 0:
            pass
        else:
            nsr_mean_sum_abs += np.log(tmpsum)
            valid_layer_count += 1  # 新增：累加有效层数


    # 新增：如果有效层数 > 0，则除以有效层数做平均
    if valid_layer_count > 0:
        return nsr_mean_sum_abs / valid_layer_count
        # return nsr_mean_sum_abs
    else:
        return 0.0

def compute_zico_score(model, train_batches, loss_fn, device: torch.device):
    """使用安全的 ZiCo 计算实现，替代原始库函数。"""
    grad_dict = {}
    model.train()
    model.to(device)
    
    # 确保 loss_fn 在设备上
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn.to(device)
    
    for i, batch in enumerate(train_batches):
        model.zero_grad()
        data, label = batch
        data = data.to(device)
        label = label.to(device)
        
        # 前向传播
        logits = model(data)
        # 计算损失
        loss = loss_fn(logits, label)
        # 反向传播
        loss.backward()
        
        # 获取梯度（使用安全版本）
        grad_dict = getgrad_safe(model, grad_dict, i)

        # 极速显存释放
        model.zero_grad(set_to_none=True) 
        
    # 计算分数
    res = caculate_zico_safe(grad_dict)
    
    # 显式释放 CPU 内存（帮助 GC 更快回收）
    del grad_dict
    
    return res

# === 新增：自定义的 ZiCo 计算逻辑，修复梯度为 None 的问题 ===


def evaluate_task(task: str, ss_name: str, args):
    """在单个任务上评估 ZiCo 与真值的相关性。"""
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
    zico_scores = []  # 存储 ZiCo
    arch_hashes = []  # 存储每个架构的哈希（op_indices）
    start = time.time()  # 计时开始
    
    # 使用 tqdm 显示进度条
    for i, graph in enumerate(tqdm(samples, desc=f"[{task}] 评估架构", unit="arch")):
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
            zico_scores.append(0.0)  # 仅占位
            continue  # 跳过计算
        
        # === 关键修改：在这里才实例化模型（延迟实例化） ===
        graph.parse()  # 实例化模型参数
        model = graph.to(args.device)  # 取具体模型并放设备
        model.train()  # 训练模式
        
        # 计算 ZiCo 分数
        try:
            zc = compute_zico_score(model, train_batches, loss_fn, args.device)  # 计算 ZiCo
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"!!! OOM detected at sample {i}. Trying to clear cache and retry...")
                torch.cuda.empty_cache()
                zc = 0.0 # OOM 时给 0 分，或者你可以选择跳过
            else:
                raise e
                
        zico_scores.append(zc)  # 记录分数
        
        # === 显存和内存清理（彻底释放） ===
        model.cpu()  # 关键：必须移回 CPU
        del model  # 删除模型引用
        
        # 关键新增：删除 graph 对象，释放其持有的模型参数
        # 因为 graph.parse() 后，graph 内部持有实例化的模型参数
        del graph  # 删除 graph 引用
        samples[i] = None  # 将列表中的引用也置为 None，确保 GC 可以回收
        
        torch.cuda.empty_cache()  # 清理显存碎片
        
        # === 强制 GC ===
        import gc
        gc.collect()

    elapsed = time.time() - start  # 总耗时
    # 相关性评估
    corr = compute_scores(ytest=gt_scores, test_pred=zico_scores)  # 计算相关性
    return {
        "task": task,  # 任务名
        "kendalltau": corr.get("kendalltau"),  # Kendall Tau
        "spearman": corr.get("spearman"),  # Spearman
        "time": elapsed,  # 用时
        "gt": gt_scores,  # 真值列表（与架构顺序一致）
        "pred": zico_scores,  # 预测列表（与架构顺序一致）
        "arch_hash": arch_hashes,  # 架构哈希列表（每个元素通常是长度为 6 的 op_indices 列表）
    }  # 返回结果字典


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run ZiCo on TransNAS-Bench-101")  # 创建解析器
    parser.add_argument("--tasks", nargs="+", default=["autoencoder", "segmentsemantic", "normal"], help="任务列表")  # 任务参数
    parser.add_argument("--search_space", choices=["micro", "macro"], default="micro", help="搜索空间类型")  # 空间选择
    parser.add_argument("--data_root", type=str, default=str(NASLIB_ROOT / "data"), help="数据根路径")  # 数据路径
    parser.add_argument("--num_samples", type=int, default=20, help="采样架构数量")  # 采样数量
    parser.add_argument("--batch_size", type=int, default=128, help="DataLoader 的 batch 大小")  # batch 大小
    parser.add_argument("--maxbatch", type=int, default=2, help="截断的 batch 数")  # 截断 batch 数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")  # 种子
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")  # 设备选择
    parser.add_argument("--dry_run", action="store_true", help="仅跑数据管线，不计算 ZiCo")  # dry run
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
