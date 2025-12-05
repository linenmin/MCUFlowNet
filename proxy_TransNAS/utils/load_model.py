import argparse  # 解析命令行参数
import random  # 控制随机性
import sys  # 调整 sys.path
from pathlib import Path  # 处理路径
import itertools  # 截断迭代
import importlib.util  # 动态加载模块

import torch  # 张量计算
import numpy as np  # 数值运算

from naslib import utils  # NASLib 工具

# ---------------- 路径与全局缓存 ----------------
CURRENT_DIR = Path(__file__).resolve().parent  # 当前文件夹
ROOT_DIR = CURRENT_DIR.parent.parent  # 项目根目录（MCUFlowNet）
NASLIB_ROOT = ROOT_DIR / "NASLib"  # NASLib 根路径
sys.path.insert(0, str(ROOT_DIR))  # 将项目根加入路径
sys.path.insert(0, str(NASLIB_ROOT))  # 将 NASLib 加入路径

_GLOBAL_TRANSNASBENCH_API = None  # 缓存 TransNASBenchAPI

# NASLib 别名引用
get_train_val_loaders = utils.get_train_val_loaders  # 数据加载函数
compute_scores = utils.compute_scores  # 评分计算函数


def load_transbench_classes():  # 加载搜索空间类
    """加载 TransNASBench 搜索空间类，避免额外依赖。"""
    graph_path = NASLIB_ROOT / "naslib" / "search_spaces" / "transbench101" / "graph.py"  # graph.py 路径
    spec = importlib.util.spec_from_file_location("transbench_graph", graph_path)  # 创建加载规范
    module = importlib.util.module_from_spec(spec)  # 创建模块对象
    spec.loader.exec_module(module)  # 执行模块
    return module.TransBench101SearchSpaceMicro, module.TransBench101SearchSpaceMacro, module  # 返回类与模块


def load_transbench_api(data_root: Path, task: str):  # 加载并缓存 API
    """加载并缓存 TransNASBenchAPI，减少重复 IO。"""
    global _GLOBAL_TRANSNASBENCH_API  # 使用全局缓存
    api_path = NASLIB_ROOT / "naslib" / "search_spaces" / "transbench101" / "api.py"  # api.py 路径
    spec = importlib.util.spec_from_file_location("transbench_api", api_path)  # 创建加载规范
    module = importlib.util.module_from_spec(spec)  # 创建模块对象
    spec.loader.exec_module(module)  # 执行模块
    TransNASBenchAPI = module.TransNASBenchAPI  # 取出 API 类

    candidate = data_root / "transnas-bench_v10141024.pth"  # 用户指定路径
    if not candidate.exists():  # 若用户路径不存在
        candidate = NASLIB_ROOT / "naslib" / "data" / "transnas-bench_v10141024.pth"  # 回退默认路径
    assert candidate.exists(), f"缺少 transnas-bench_v10141024.pth，检查 {candidate}"  # 确保文件存在

    if _GLOBAL_TRANSNASBENCH_API is None:  # 若未缓存
        _GLOBAL_TRANSNASBENCH_API = TransNASBenchAPI(str(candidate))  # 载入并缓存

    return {"api": _GLOBAL_TRANSNASBENCH_API, "task": task}  # 返回 API 字典


def build_config(data_root: Path, dataset: str, batch_size: int, seed: int):  # 构建最小配置
    """构建最小配置对象以复用 NASLib 数据加载。"""
    search_cfg = argparse.Namespace(  # 搜索配置
        seed=seed,  # 随机种子
        batch_size=batch_size,  # batch 大小
        train_portion=0.7,  # 训练集占比
    )
    config = argparse.Namespace(  # 顶层配置
        data=str(data_root),  # 数据根路径
        dataset=dataset,  # 数据集名称
        search=search_cfg,  # 搜索配置
    )
    return config  # 返回配置


def make_train_loader(task: str, data_root: Path, batch_size: int, seed: int):  # 生成训练 loader
    """生成指定任务的训练 DataLoader。"""
    config = build_config(data_root, task, batch_size, seed)  # 构建配置
    train_loader, _, _, _, _ = get_train_val_loaders(config)  # 调用 NASLib 加载
    return train_loader  # 返回训练 loader


def truncate_loader(loader, max_batch: int):  # 截断 loader
    """截断 DataLoader，只保留前 max_batch 个 batch。"""
    return list(itertools.islice(iter(loader), max_batch))  # 使用 islice 截断


def get_metric_name(task: str):  # 获取 metric 名
    """根据任务名称返回正确的 metric 名。"""
    if task in ["autoencoder", "normal"]:  # 自编码与法线
        return "valid_ssim"  # 使用 SSIM
    if task == "segmentsemantic":  # 语义分割
        return "valid_acc"  # 使用准确率
    if task == "room_layout":  # 房间布局
        return "valid_neg_loss"  # 使用负损失
    return "valid_top1"  # 其他任务默认 top1


def sample_architecture_identifiers(ss, dataset_api, num_samples: int, seed: int):  # 采样架构标识
    """采样若干不重复的架构标识（op_indices），便于低内存存储。"""
    random.seed(seed)  # 固定随机
    torch.manual_seed(seed)  # 固定 torch 随机

    arch_identifiers = []  # 存放架构标识
    seen_hashes = set()  # 已见哈希集合
    max_attempts = num_samples * 10  # 最大尝试次数
    attempts = 0  # 当前尝试计数

    print(f"正在采样 {num_samples} 个不重复的架构...", end="", flush=True)  # 提示信息

    while len(arch_identifiers) < num_samples and attempts < max_attempts:  # 采样循环
        attempts += 1  # 增加计数
        temp_graph = ss.clone()  # 克隆搜索空间
        temp_graph.sample_random_architecture(dataset_api=dataset_api)  # 随机采样
        try:
            arch_hash = tuple(temp_graph.get_hash())  # 获取哈希
        except Exception:
            del temp_graph  # 释放资源
            continue  # 跳过错误

        if arch_hash in seen_hashes:  # 检查重复
            del temp_graph  # 释放资源
            continue  # 跳过重复

        seen_hashes.add(arch_hash)  # 记录哈希
        arch_identifiers.append(arch_hash)  # 保存标识
        del temp_graph  # 释放资源

    if len(arch_identifiers) < num_samples:  # 采样不足
        print(f"\n警告：只采到 {len(arch_identifiers)} 个（目标 {num_samples}），可能搜索空间太小")  # 提示警告
    else:
        print(f" 完成（尝试 {attempts} 次）")  # 完成提示

    return arch_identifiers  # 返回列表


def set_op_indices_from_str(ss_name: str, graph, arch_str: str):  # 从架构字符串写入 op_indices
    """根据搜索空间名称解析架构字符串并写入图的 op_indices。"""  # 函数说明
    if ss_name == "micro":  # 微搜索空间
        parts = arch_str.split("-")  # 拆分字符串
        config_part = parts[-1]  # 取配置段
        config_parts = config_part.split("_")  # 再拆分
        op_indices = []  # 初始化列表
        op_indices.append(int(config_parts[0]))  # stem 操作
        for digit in config_parts[1]:  # 遍历第二段
            op_indices.append(int(digit))  # 加入索引
        for digit in config_parts[2]:  # 遍历第三段
            op_indices.append(int(digit))  # 加入索引
        graph.set_op_indices(op_indices)  # 设置索引
    else:  # 宏搜索空间
        parts = arch_str.split("-")  # 拆分字符串
        ops_string = parts[1]  # 取操作串
        op_indices = [int(d) for d in ops_string]  # 转为列表
        while len(op_indices) < 6:  # 长度补齐
            op_indices.append(0)  # 用 0 填充
        graph.set_op_indices(op_indices)  # 设置索引
    return graph  # 返回图对象


def select_architectures_by_percentile(api_dict: dict, ss_name: str, task: str, start_percent: float, end_percent: float):  # 百分位选架构
    """按照 GT 百分位区间选择架构字符串列表。"""
    api = api_dict["api"]  # 取出 API
    all_archs = api.all_arch_dict[ss_name]  # 获取所有架构
    metric_name = get_metric_name(task)  # 选择 metric

    arch_scores = []  # 存放架构与分数
    for arch_str in all_archs:  # 遍历架构
        try:
            gt_score = api.get_single_metric(arch_str, task, metric_name, mode="final")  # 查询 GT
            arch_scores.append((arch_str, gt_score))  # 记录结果
        except Exception:
            continue  # 忽略异常

    total_archs = len(arch_scores)  # 总数量
    if total_archs == 0:  # 若无数据
        print("警告：未找到有效架构")  # 输出警告
        return []  # 返回空列表

    arch_scores_sorted = sorted(arch_scores, key=lambda x: x[1], reverse=True)  # 按分数排序
    start_idx = int(total_archs * start_percent / 100)  # 起始索引
    end_idx = int(total_archs * end_percent / 100)  # 结束索引
    start_idx = max(0, min(start_idx, total_archs - 1))  # 规范起点
    end_idx = max(start_idx + 1, min(end_idx, total_archs))  # 规范终点

    selected_archs = arch_scores_sorted[start_idx:end_idx]  # 选取区间
    if len(selected_archs) == 0:  # 若区间为空
        print("警告：当前区间无架构")  # 输出提示
        return []  # 返回空

    return [arch_str for arch_str, _ in selected_archs]  # 返回架构字符串
