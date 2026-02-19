"""Analyze subnet distribution and export data files only."""  # 脚本用于分析子网分布并只导出数据文件

import argparse
import csv
import itertools
import json
import random
import shutil
import time
from datetime import datetime, timezone
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
from tqdm import tqdm  # 进度条库

from code.utils.json_io import write_json  # 自定义 JSON 写入工具

NUM_BLOCKS = 9  # 网络结构一共有 9 个可搜索 block
ARCH_SPACE_SIZE = 3 ** NUM_BLOCKS  # 搜索空间大小：每个 block 3 种选择
PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)  # 需要统计的分位点


def _iter_all_arch_codes(num_blocks: int = NUM_BLOCKS) -> Iterable[List[int]]:
    """Iterate full architecture search space."""  # 遍历整个架构搜索空间
    for code in itertools.product((0, 1, 2), repeat=num_blocks):  # 生成所有 3^num_blocks 的组合
        yield [int(item) for item in code]  # 转成 int 列表后逐个返回


def _dedup_codes(codes: Sequence[Sequence[int]]) -> List[List[int]]:
    """Keep first occurrence order while dropping duplicates."""  # 去重但保持首次出现顺序
    seen = set()  # 记录已出现的架构
    out: List[List[int]] = []  # 输出列表
    for code in codes:  # 遍历输入架构序列
        key = tuple(int(item) for item in code)  # 转成不可变元组方便放入 set
        if key in seen:  # 如果已出现则跳过
            continue
        seen.add(key)  # 标记为已出现
        out.append([int(item) for item in code])  # 保存原始 int 列表
    return out  # 返回去重后的列表


def sample_arch_pool(
    num_arch_samples: int,
    seed: int,
    include_eval_pool: bool,
    eval_pool: Sequence[Sequence[int]],  # 固定评估池（例如 12 个子网）
) -> List[List[int]]:
    """Sample unique architecture codes from 3^9 space."""  # 从 3^9 空间中采样唯一架构
    sample_size = int(num_arch_samples)  # 采样数量转成 int
    if sample_size <= 0:  # 校验采样数量
        raise ValueError("num_arch_samples must be > 0")  # 非法时抛出异常

    if sample_size >= ARCH_SPACE_SIZE:  # 如果采样数超过空间大小
        return list(_iter_all_arch_codes())  # 直接返回全空间

    rng = random.Random(int(seed))  # 使用给定种子创建随机数生成器
    pool: List[List[int]] = []  # 保存最终采样的架构池
    if include_eval_pool:  # 如果需要包含固定评估池
        pool.extend(_dedup_codes(eval_pool))  # 先把评估池去重加入
    if len(pool) >= sample_size:  # 如果此时数量已经够了
        return pool[:sample_size]  # 截断到目标数量

    seen = {tuple(code) for code in pool}  # 已有架构集合，用于去重
    while len(pool) < sample_size:  # 循环直到采样数量满足
        candidate = [rng.randint(0, 2) for _ in range(NUM_BLOCKS)]  # 随机生成一个 9 维架构
        key = tuple(candidate)  # 转为元组做去重 key
        if key in seen:  # 已经存在则跳过
            continue
        seen.add(key)  # 记录新 key
        pool.append(candidate)  # 加入采样池
    return pool  # 返回最终采样池


def compute_complexity_scores(arch_code: Sequence[int]) -> Dict[str, float]:
    """Compute unified-direction complexity proxy scores."""  # 计算统一方向的复杂度代理分数
    if len(arch_code) != NUM_BLOCKS:  # 校验架构长度
        raise ValueError("arch_code length must be 9")  # 长度不为 9 抛异常
    code = [int(item) for item in arch_code]  # 转成 int 列表
    depth_score = float(sum(code[:4]))  # 前四位越大表示骨干越重。
    kernel_light_score = float(sum(code[4:]))  # 后五位越大表示头部越轻。
    kernel_heavy_score = float(sum(2 - item for item in code[4:]))  # 统一成越大越重方向。
    total_score = float(depth_score + kernel_heavy_score)  # 总复杂度代理分数。
    return {
        "depth_score": depth_score,  # 骨干复杂度
        "kernel_light_score": kernel_light_score,  # 头部轻量化程度
        "kernel_heavy_score": kernel_heavy_score,  # 头部“重”程度（统一方向）
        "complexity_score": total_score,  # 总体复杂度分数
    }


def _safe_float(value: Any) -> Optional[float]:
    """Convert value to float with best effort."""  # 尝试安全地转换成 float
    if value is None:  # None 直接返回 None
        return None
    try:
        number = float(value)  # 尝试转成 float
    except Exception:
        return None  # 转换失败返回 None
    if np.isfinite(number):  # 只接受有限数值
        return float(number)
    return None  # NaN 或 inf 返回 None


def _safe_fps(inference_ms: Optional[float]) -> Optional[float]:
    """Convert inference time(ms) to FPS."""  # 将推理时间（毫秒）转换为 FPS
    time_ms = _safe_float(inference_ms)  # 先安全转成 float
    if time_ms is None or time_ms <= 0.0:  # 非法或小于等于 0 直接返回 None
        return None
    return float(1000.0 / time_ms)  # 1000 毫秒 / 单帧时间 = FPS


def _compute_metric_summary(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    """Compute robust summary for one metric."""  # 计算某个指标的一组统计量
    cleaned = [float(item) for item in values if _safe_float(item) is not None]  # 过滤掉无效值
    total_count = int(len(values))  # 原始总数
    valid_count = int(len(cleaned))  # 有效数量
    if valid_count <= 0:  # 如果没有有效值
        return {
            "count_total": total_count,  # 总数
            "count_valid": 0,  # 有效数为 0
            "count_invalid": total_count,  # 全部无效
        }
    arr = np.asarray(cleaned, dtype=np.float64)  # 转成 numpy 数组方便统计
    summary: Dict[str, Any] = {
        "count_total": total_count,  # 总数
        "count_valid": valid_count,  # 有效数
        "count_invalid": int(total_count - valid_count),  # 无效数
        "mean": float(np.mean(arr)),  # 均值
        "std": float(np.std(arr)),  # 标准差
        "min": float(np.min(arr)),  # 最小值
        "max": float(np.max(arr)),  # 最大值
        "median": float(np.median(arr)),  # 中位数
    }
    for pct in PERCENTILES:  # 遍历需要的分位点
        summary[f"p{pct:02d}"] = float(np.percentile(arr, pct))  # 保存对应百分位数
    return summary  # 返回统计结果


def _to_arch_text(arch_code: Sequence[int]) -> str:
    """Format arch code for CSV/export."""  # 将架构编码转成字符串形式
    return ",".join(str(int(item)) for item in arch_code)  # 用逗号连接每一位


def _save_records_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    """Save sampled subnet records to CSV."""  # 将采样子网的记录保存到 CSV
    fields = [
        "sample_index",  # 采样索引
        "arch_code",  # 架构编码字符串
        "epe",  # 端点误差 EPE
        "depth_score",  # 深度得分
        "kernel_light_score",  # 头部轻量化得分
        "kernel_heavy_score",  # 头部重度得分
        "complexity_score",  # 总体复杂度分数
        "sram_peak_mb",  # Vela 估算的峰值 SRAM（MB）
        "inference_ms",  # Vela 估算的推理时间（毫秒）
        "fps",  # 由推理时间换算的 FPS
        "vela_status",  # Vela 运行状态
        "vela_error",  # Vela 错误信息
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:  # 打开 CSV 文件准备写入
        writer = csv.DictWriter(handle, fieldnames=fields)  # 使用字典写入器
        writer.writeheader()  # 先写表头
        for row in records:  # 遍历每条记录
            writer.writerow({key: row.get(key, "") for key in fields})  # 只写需要的字段


def _save_ranking_csv(path: Path, ranking: Sequence[Dict[str, Any]]) -> None:
    """Save sorted ranking by EPE to CSV."""  # 将按 EPE 排序的排名写入 CSV
    fields = ["rank", "sample_index", "arch_code", "epe", "complexity_score"]  # 导出的字段列表
    with path.open("w", encoding="utf-8", newline="") as handle:  # 打开 CSV 文件
        writer = csv.DictWriter(handle, fieldnames=fields)  # 创建写入器
        writer.writeheader()  # 写入表头
        for row in ranking:  # 遍历排序后的行
            writer.writerow({key: row.get(key, "") for key in fields})  # 写入一行


def _save_vela_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    """Save Vela-only metrics to CSV."""  # 单独保存 Vela 相关指标到 CSV
    fields = [
        "sample_index",  # 采样索引
        "arch_code",  # 架构编码
        "sram_peak_mb",  # 峰值 SRAM
        "inference_ms",  # 推理时间（毫秒）
        "fps",  # FPS
        "vela_status",  # Vela 运行状态
        "vela_error",  # 错误信息
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:  # 打开目标文件
        writer = csv.DictWriter(handle, fieldnames=fields)  # 创建 CSV 写入器
        writer.writeheader()  # 写入表头
        for row in records:  # 遍历记录
            writer.writerow({key: row.get(key, "") for key in fields})  # 写入一行


def _evaluate_arch_pool(
    config: Dict[str, Any],
    checkpoint_prefix: Path,
    arch_pool: List[List[int]],
    bn_recal_batches: int,
    batch_size: int,
    eval_batches_per_arch: int,
    num_workers: int,
    cpu_only: bool,  # 是否只用 CPU 进行评估
) -> Dict[str, Any]:
    """Evaluate sampled subnets and return per-arch EPE."""  # 评估采样的子网并返回每个子网的 EPE
    from code.data.dataloader_builder import build_fc2_provider  # 导入数据提供器构建函数
    from code.nas.supernet_eval import (
        _build_eval_graph,  # 构建评估计算图
        _run_eval_pool,  # 单进程评估子网
        _run_eval_pool_parallel,  # 多进程评估子网
    )
    import tensorflow as tf  # 导入 TensorFlow

    workers_used = min(max(1, int(num_workers)), max(1, len(arch_pool)))  # 实际使用的 worker 数量
    if workers_used <= 1:  # 单 worker 情况（单进程评估）
        if cpu_only:  # 若强制 CPU 运行
            try:
                tf.config.set_visible_devices([], "GPU")  # 屏蔽 GPU 设备
            except Exception:
                pass  # 失败则忽略
        tf.compat.v1.disable_eager_execution()  # 关闭 eager 执行，使用 TF1 图模式
        tf.compat.v1.reset_default_graph()  # 清空默认图
        graph_obj = _build_eval_graph(config=config, batch_size=batch_size)  # 构建评估图
        train_provider = build_fc2_provider(config=config, split="train", seed_offset=0, provider_mode="eval")  # 训练集 provider
        val_provider = build_fc2_provider(config=config, split="val", seed_offset=999, provider_mode="eval")  # 验证集 provider
        if len(train_provider) == 0:  # 如果训练样本数为 0
            raise RuntimeError(f"train sample count is 0; source={train_provider.source_dir}")  # 抛异常提示数据源
        if len(val_provider) == 0:  # 如果验证样本数为 0
            raise RuntimeError(f"val sample count is 0; source={val_provider.source_dir}")  # 抛异常提示
        with tf.compat.v1.Session() as sess:  # 创建 TF1 会话
            sess.run(tf.compat.v1.global_variables_initializer())  # 初始化变量
            graph_obj["saver"].restore(sess, str(checkpoint_prefix))  # 从 checkpoint 恢复权重
            metrics = _run_eval_pool(
                sess=sess,  # TF 会话
                graph_obj=graph_obj,  # 图对象
                train_provider=train_provider,  # 训练数据 provider（用于 BN 重校准）
                val_provider=val_provider,  # 验证数据 provider
                eval_pool=arch_pool,  # 要评估的架构池
                bn_recal_batches=int(bn_recal_batches),  # BN 重校准批次数
                batch_size=int(batch_size),  # batch 大小
                eval_batches_per_arch=int(eval_batches_per_arch),  # 每个子网评估多少个 batch
            )
        metrics["num_workers_used"] = 1  # 记录实际使用的 worker 数
        return metrics  # 返回指标

    # 多 worker 情况，调用并行评估入口
    metrics = _run_eval_pool_parallel(
        config=config,  # 配置
        checkpoint_prefix=checkpoint_prefix,  # 模型 checkpoint 前缀
        eval_pool=arch_pool,  # 架构池
        bn_recal_batches=int(bn_recal_batches),  # BN 重校准批数
        batch_size=int(batch_size),  # batch 大小
        eval_batches_per_arch=int(eval_batches_per_arch),  # 每个子网评估 batch 数
        num_workers=int(workers_used),  # 实际 worker 数
        cpu_only=bool(cpu_only),  # 是否只用 CPU
    )
    metrics["num_workers_used"] = int(workers_used)  # 在返回结果中记录 worker 数
    return metrics  # 返回评估结果


def _build_tflite_for_arch(
    config: Dict[str, Any],
    checkpoint_prefix: Path,
    arch_code: Sequence[int],
    tflite_path: Path,
    rep_dataset_samples: int,
    quantize_int8: bool,  # 是否导出 INT8 量化模型
) -> None:
    """Export one fixed-arch subnet to TFLite."""  # 将固定架构子网导出为 TFLite 模型
    import tensorflow as tf  # 导入 TensorFlow

    from code.network.MultiScaleResNet_supernet import MultiScaleResNetSupernet  # 导入 supernet 网络定义

    data_cfg = config.get("data", {})  # 读取数据相关配置
    input_h = int(data_cfg.get("input_height", 180))  # 输入高度，默认 180
    input_w = int(data_cfg.get("input_width", 240))  # 输入宽度，默认 240
    flow_channels = int(data_cfg.get("flow_channels", 2))  # 输入光流通道数
    pred_channels = int(flow_channels * 2)  # 预测输出通道数（估计双向）

    tf.compat.v1.disable_eager_execution()  # 保持 TF1 图模式一致。
    graph = tf.Graph()  # 创建一个独立的 TF 图
    with graph.as_default():  # 在该图上下文中定义算子
        input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, input_h, input_w, 6], name="Input")  # 输入占位符
        arch_const = tf.constant([int(v) for v in arch_code], dtype=tf.int32, name="ArchCodeConst")  # 固定架构常量
        is_training_const = tf.constant(False, dtype=tf.bool, name="IsTrainingConst")  # 训练/测试标志常量
        model = MultiScaleResNetSupernet(
            input_ph=input_ph,  # 输入张量
            arch_code_ph=arch_const,  # 架构编码
            is_training_ph=is_training_const,  # 是否训练
            num_out=pred_channels,  # 输出通道数
            init_neurons=32,  # 初始通道数
            expansion_factor=2.0,  # 通道扩展倍数
        )
        preds = model.build()  # 构建网络，得到多尺度输出列表
        output_tensor = preds[-1]  # 取最后一个尺度作为最终输出
        saver = tf.compat.v1.train.Saver(max_to_keep=1)  # 用于恢复权重的 Saver

        with tf.compat.v1.Session(graph=graph) as sess:  # 创建会话绑定到该图
            sess.run(tf.compat.v1.global_variables_initializer())  # 初始化变量
            saver.restore(sess, str(checkpoint_prefix))  # 从 checkpoint 恢复权重
            converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [input_ph], [output_tensor])  # 创建 TFLite 转换器
            if quantize_int8:  # 如果需要 INT8 量化
                converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 打开默认优化
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,  # 支持 INT8 内置算子
                    tf.lite.OpsSet.TFLITE_BUILTINS,  # 同时也允许 float 算子
                ]
                converter.inference_input_type = tf.int8  # 推理输入类型 INT8
                converter.inference_output_type = tf.int8  # 推理输出类型 INT8
                rng = np.random.default_rng(seed=2026)  # 随机数生成器用于代表性数据

                def representative_dataset_gen():
                    """代表性数据生成器，用于量化校准。"""  # 为量化提供样本
                    for _ in range(max(1, int(rep_dataset_samples))):  # 生成指定数量样本
                        sample = rng.random((1, input_h, input_w, 6)).astype(np.float32)  # 随机生成输入
                        yield [sample]  # 以列表形式返回

                converter.representative_dataset = representative_dataset_gen  # 注册代表性数据集
            tflite_model = converter.convert()  # 执行 TFLite 转换，得到二进制模型
    tflite_path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
    tflite_path.write_bytes(tflite_model)  # 将 TFLite 模型写入文件


def _run_vela_for_arch(
    tflite_path: Path,
    output_dir: Path,
    vela_mode: str,
    vela_optimise: str,
    vela_silent: bool,
) -> Dict[str, Any]:
    """Run Vela and return parsed metrics."""  # 调用 Vela 编译器并返回解析后的指标
    def _resolve_run_vela() -> Callable[..., Any]:
        """Load run_vela with module import fallback."""  # 先走包导入，失败后走文件路径兜底
        try:
            from tests.vela.vela_compiler import run_vela as imported_run_vela

            return imported_run_vela
        except Exception as import_exc:
            vela_file = Path(__file__).resolve().parents[2] / "tests" / "vela" / "vela_compiler.py"
            if not vela_file.exists():
                raise ImportError(
                    "failed to import tests.vela.vela_compiler.run_vela and fallback file not found: "
                    f"{vela_file}"
                ) from import_exc
            spec = importlib_util.spec_from_file_location("edgeflownas_tests_vela_compiler", str(vela_file))
            if spec is None or spec.loader is None:
                raise ImportError(f"failed to build module spec for {vela_file}") from import_exc
            module = importlib_util.module_from_spec(spec)
            spec.loader.exec_module(module)
            run_vela_func = getattr(module, "run_vela", None)
            if run_vela_func is None:
                raise ImportError(f"run_vela not found in fallback module: {vela_file}")
            return run_vela_func

    try:
        run_vela = _resolve_run_vela()  # 导入 run_vela（支持路径兜底）
    except Exception as exc:
        raise ImportError(  # 导入失败时抛出更明确的错误
            "failed to import tests.vela.vela_compiler.run_vela; "
            "please ensure EdgeFlowNAS/tests/vela exists"
        ) from exc

    sram_mb, inference_ms = run_vela(
        str(tflite_path),  # TFLite 模型路径
        mode=str(vela_mode),  # Vela 模式（basic / verbose）
        output_dir=str(output_dir),  # Vela 输出目录
        optimise=str(vela_optimise),  # 优化目标（性能 / 体积）
        silent=bool(vela_silent),  # 是否静默模式
    )
    sram_peak_mb = _safe_float(sram_mb)  # 安全转换 SRAM 数值
    inference_ms = _safe_float(inference_ms)  # 安全转换推理时间
    fps = _safe_fps(inference_ms)  # 根据推理时间计算 FPS
    status = "ok" if sram_peak_mb is not None and inference_ms is not None and fps is not None else "fail"  # 判断是否成功
    error_text = "" if status == "ok" else "missing_vela_metrics"  # 失败时给出简单错误标记
    return {
        "sram_peak_mb": sram_peak_mb,  # SRAM 峰值（MB）
        "inference_ms": inference_ms,  # 推理时间（毫秒）
        "fps": fps,  # FPS
        "vela_status": status,  # 状态
        "vela_error": error_text,  # 错误信息
    }


def _collect_vela_metrics(
    config: Dict[str, Any],
    checkpoint_prefix: Path,
    arch_pool: List[List[int]],
    analysis_dir: Path,
    vela_mode: str,
    vela_optimise: str,
    vela_silent: bool,
    vela_limit: Optional[int],
    rep_dataset_samples: int,
    quantize_int8: bool,
    keep_artifacts: bool,
) -> Dict[int, Dict[str, Any]]:
    """Collect Vela metrics for sampled subnets."""  # 为采样子网批量收集 Vela 指标
    results: Dict[int, Dict[str, Any]] = {}  # 保存每个样本索引到 Vela 结果的映射
    limit = len(arch_pool) if vela_limit is None else max(0, min(len(arch_pool), int(vela_limit)))  # 实际 Vela 运行的最大样本数
    vela_root = analysis_dir / "vela_tmp"  # Vela 临时目录根路径
    vela_root.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    ok_count = 0  # 统计 Vela 成功数
    fail_count = 0  # 统计 Vela 失败数

    with tqdm(total=limit, desc="subnet vela", unit="arch") as progress:  # 进度条显示 Vela 进度
        for sample_index, arch_code in enumerate(arch_pool[:limit]):  # 遍历需要跑 Vela 的子网
            arch_tag = f"arch_{sample_index:04d}"  # 为每个子网生成标签名
            arch_dir = vela_root / arch_tag  # 当前子网对应的临时目录
            tflite_path = arch_dir / f"{arch_tag}.tflite"  # 当前子网的 TFLite 文件路径
            arch_dir.mkdir(parents=True, exist_ok=True)  # 创建该目录
            item = {
                "sram_peak_mb": None,  # 默认 SRAM 为空
                "inference_ms": None,  # 默认推理时间为空
                "fps": None,  # 默认 FPS 为空
                "vela_status": "fail",  # 默认视为失败，成功后再覆盖
                "vela_error": "",  # 默认错误信息为空
            }
            try:
                _build_tflite_for_arch(
                    config=config,  # 配置
                    checkpoint_prefix=checkpoint_prefix,  # checkpoint 路径前缀
                    arch_code=arch_code,  # 当前子网架构编码
                    tflite_path=tflite_path,  # 导出的 TFLite 路径
                    rep_dataset_samples=rep_dataset_samples,  # 代表性数据样本数
                    quantize_int8=quantize_int8,  # 是否 INT8 量化
                )
                vela_item = _run_vela_for_arch(
                    tflite_path=tflite_path,  # TFLite 模型
                    output_dir=arch_dir,  # 当前子网的 Vela 输出目录
                    vela_mode=vela_mode,  # Vela 模式
                    vela_optimise=vela_optimise,  # 优化策略
                    vela_silent=vela_silent,  # 是否静默
                )
                item.update(vela_item)  # 合并 Vela 返回的指标
            except Exception as exc:  # 捕获任意异常
                item["vela_status"] = "fail"  # 标记失败
                item["vela_error"] = str(exc)  # 记录错误信息
            if item["vela_status"] == "ok":  # 如果 Vela 运行成功
                ok_count += 1  # 成功计数加一
            else:
                fail_count += 1  # 失败计数加一
            results[int(sample_index)] = item  # 把结果放入字典
            if not keep_artifacts:  # 如果不保留中间产物
                shutil.rmtree(arch_dir, ignore_errors=True)  # 删除当前子网目录
            progress.update(1)  # 更新进度条
            progress.set_postfix(ok=ok_count, fail=fail_count)  # 在进度条尾部显示成功/失败数

    if not keep_artifacts and vela_root.exists():  # 若不保留产物且总临时目录存在
        shutil.rmtree(vela_root, ignore_errors=True)  # 删除整个 Vela 临时根目录
    return results  # 返回所有子网的 Vela 结果


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""  # 构建命令行参数解析器
    parser = argparse.ArgumentParser(description="analyze subnet distribution under best/last checkpoint")  # 描述脚本用途
    parser.add_argument("--config", required=True, help="path to supernet config yaml")  # 超网配置文件路径
    parser.add_argument("--checkpoint_type", choices=["best", "last"], default="best", help="checkpoint type")  # 选择使用最佳还是最后一个 checkpoint
    parser.add_argument("--num_arch_samples", type=int, default=512, help="number of sampled subnets")  # 采样子网数量
    parser.add_argument("--sample_seed", type=int, default=None, help="seed for subnet sampling")  # 子网采样随机种子
    parser.add_argument("--exclude_eval_pool", action="store_true", help="exclude fixed eval_pool_12 from samples")  # 是否排除固定评估池架构

    parser.add_argument("--bn_recal_batches", type=int, default=None, help="override BN recalibration batches")  # 覆盖 BN 重校准批数
    parser.add_argument("--eval_batches_per_arch", type=int, default=None, help="override val eval batches per arch")  # 覆盖验证时每个子网评估批数
    parser.add_argument("--batch_size", type=int, default=None, help="override eval batch size")  # 覆盖评估 batch 大小
    parser.add_argument("--num_workers", type=int, default=1, help="parallel workers for arch eval")  # 子网评估并行 worker 数
    parser.add_argument("--cpu_only", action="store_true", help="force CPU-only eval")  # 强制只用 CPU

    parser.add_argument("--enable_vela", action="store_true", help="enable Vela benchmark for sampled subnets")  # 是否运行 Vela 评估
    parser.add_argument("--vela_mode", choices=["basic", "verbose"], default="basic", help="Vela mode")  # Vela 输出模式
    parser.add_argument("--vela_optimise", choices=["Performance", "Size"], default="Size", help="Vela optimise policy")  # Vela 优化策略
    parser.add_argument("--vela_limit", type=int, default=None, help="limit Vela runs by sample count")  # 限制参与 Vela 的子网数量
    parser.add_argument("--vela_rep_dataset_samples", type=int, default=3, help="rep-dataset samples for INT8 export")  # INT8 量化代表性数据样本数
    parser.add_argument("--vela_float32", action="store_true", help="export float32 TFLite instead of INT8")  # 是否导出 float32 TFLite 而非 INT8
    parser.add_argument("--vela_keep_artifacts", action="store_true", help="keep Vela temp folders and tflite files")  # 是否保留 Vela 中间文件
    parser.add_argument("--vela_verbose_log", action="store_true", help="disable Vela silent mode")  # 是否打印 Vela 详细日志

    parser.add_argument("--plots", default="", help="deprecated: plotting moved to separate script")  # 已废弃：绘图相关参数
    parser.add_argument("--hist_bins", type=int, default=30, help="deprecated: plotting moved to separate script")  # 已废弃：直方图 bin 数
    parser.add_argument("--top_k", type=int, default=10, help="top/bottom K summary")  # 导出 top/bottom K 子网摘要
    parser.add_argument("--output_tag", default="", help="optional suffix for output folder")  # 输出文件夹名附加后缀

    parser.add_argument("--experiment_name", default=None, help="override runtime.experiment_name")  # 覆盖运行时实验名
    parser.add_argument("--base_path", default=None, help="override data.base_path")  # 覆盖数据根路径
    parser.add_argument("--train_dir", default=None, help="override data.train_dir")  # 覆盖训练数据路径
    parser.add_argument("--val_dir", default=None, help="override data.val_dir")  # 覆盖验证数据路径
    parser.add_argument("--train_batch_size", type=int, default=None, help="override train.batch_size in config")  # 覆盖训练 batch 大小
    parser.add_argument("--seed", type=int, default=None, help="override runtime.seed")  # 覆盖 runtime 随机种子
    return parser  # 返回解析器对象


def main() -> int:
    """CLI entry point."""  # 命令行入口函数
    parser = _build_parser()  # 构建参数解析器
    args = parser.parse_args()  # 从命令行读取参数
    from code.nas.supernet_eval import (
        _apply_cli_overrides,  # 应用命令行覆盖配置
        _build_checkpoint_paths,  # 构造 checkpoint 路径
        _find_existing_checkpoint,  # 查找实际存在的 checkpoint
        _load_checkpoint_meta,  # 读取 checkpoint 相关元信息
        _load_config,  # 加载 YAML 配置
        _load_or_build_eval_pool,  # 加载或构建评估池
        _resolve_output_dir,  # 解析输出目录
    )

    started_at = datetime.now(timezone.utc)  # 记录开始运行时间（UTC）
    start_perf = time.perf_counter()  # 性能计时起点

    config = _apply_cli_overrides(config=_load_config(args.config), args=args)  # 读取 YAML 配置并应用命令行覆盖
    runtime_cfg = config.get("runtime", {})  # 运行时配置
    train_cfg = config.get("train", {})  # 训练配置
    eval_cfg = config.get("eval", {})  # 评估配置
    seed = int(runtime_cfg.get("seed", 42))  # 获取基础随机种子

    pool_size = int(eval_cfg.get("eval_pool_size", 12))  # 评估池大小（默认 12）
    eval_pool_info = _load_or_build_eval_pool(
        output_dir=_resolve_output_dir(config=config),  # 输出目录
        seed=seed,  # 随机种子
        pool_size=pool_size,  # 池大小
    )
    include_eval_pool = not bool(args.exclude_eval_pool)  # 是否包含固定评估池
    sample_seed = int(args.sample_seed) if args.sample_seed is not None else int(seed + 9973)  # 子网采样种子（若未指定则在基础种子上偏移）
    arch_pool = sample_arch_pool(
        num_arch_samples=int(args.num_arch_samples),  # 采样子网数量
        seed=sample_seed,  # 采样随机种子
        include_eval_pool=include_eval_pool,  # 是否包含评估池
        eval_pool=eval_pool_info["pool"],  # 评估池实际列表
    )

    output_root = _resolve_output_dir(config=config)  # 解析超网训练输出目录
    ckpt_paths = _build_checkpoint_paths(experiment_dir=output_root)  # 构造 best/last checkpoint 路径
    chosen = ckpt_paths[args.checkpoint_type]  # 根据参数选择 best 或 last
    checkpoint_prefix = _find_existing_checkpoint(path_prefix=chosen)  # 寻找实际存在的 checkpoint
    if checkpoint_prefix is None:  # 若没有找到 checkpoint
        raise RuntimeError(f"checkpoint not found: {chosen}")  # 抛出异常提醒路径
    checkpoint_prefix = Path(checkpoint_prefix)  # 转成 Path 对象

    # BN 重校准批数（命令行优先，否则用配置）
    bn_recal_batches = int(args.bn_recal_batches) if args.bn_recal_batches is not None else int(eval_cfg.get("bn_recal_batches", 8))
    # 每个子网评估的 batch 数（命令行优先）
    eval_batches_per_arch = (
        int(args.eval_batches_per_arch) if args.eval_batches_per_arch is not None else int(eval_cfg.get("eval_batches_per_arch", 4))
    )
    # 评估时 batch 大小（命令行优先，否则继承训练 batch_size）
    batch_size = int(args.batch_size) if args.batch_size is not None else int(train_cfg.get("batch_size", 32))

    tag = str(args.output_tag).strip()  # 输出目录附加标签
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")  # 时间戳
    folder_name = f"{args.checkpoint_type}_{len(arch_pool)}_{stamp}" + (f"_{tag}" if tag else "")  # 运行文件夹名
    run_dir = output_root / "subnet_distribution" / folder_name  # 总运行目录
    analysis_dir = run_dir / "analysis"  # 分析结果目录
    analysis_dir.mkdir(parents=True, exist_ok=True)  # 创建分析目录

    # 评估采样子网在超网上的 EPE 表现
    metrics = _evaluate_arch_pool(
        config=config,  # 配置
        checkpoint_prefix=checkpoint_prefix,  # checkpoint 前缀
        arch_pool=arch_pool,  # 架构池
        bn_recal_batches=bn_recal_batches,  # BN 重校准批数
        batch_size=batch_size,  # batch 大小
        eval_batches_per_arch=eval_batches_per_arch,  # 每个子网评估 batch 数
        num_workers=int(args.num_workers),  # 并行 worker 数
        cpu_only=bool(args.cpu_only),  # 是否仅用 CPU
    )

    records: List[Dict[str, Any]] = []  # 保存每个子网的完整记录
    per_arch_epe = [float(item) for item in metrics.get("per_arch_epe", [])]  # 从指标中取出每个子网的 EPE 列表
    for idx, (arch_code, epe_val) in enumerate(zip(arch_pool, per_arch_epe)):  # 遍历架构与其 EPE
        score = compute_complexity_scores(arch_code=arch_code)  # 计算复杂度代理分数
        records.append(
            {
                "sample_index": int(idx),  # 采样索引
                "arch_code": _to_arch_text(arch_code),  # 架构编码字符串
                "epe": float(epe_val),  # EPE 值
                "depth_score": float(score["depth_score"]),  # 深度得分
                "kernel_light_score": float(score["kernel_light_score"]),  # 头部轻量化得分
                "kernel_heavy_score": float(score["kernel_heavy_score"]),  # 头部重度得分
                "complexity_score": float(score["complexity_score"]),  # 综合复杂度
                "sram_peak_mb": None,  # Vela 相关指标先占位
                "inference_ms": None,
                "fps": None,
                "vela_status": "skipped",  # 初始状态视为跳过 Vela
                "vela_error": "",
            }
        )

    if bool(args.enable_vela):  # 若开启 Vela 评估
        vela_map = _collect_vela_metrics(
            config=config,  # 配置
            checkpoint_prefix=checkpoint_prefix,  # checkpoint
            arch_pool=arch_pool,  # 架构池
            analysis_dir=analysis_dir,  # 分析目录
            vela_mode=str(args.vela_mode),  # Vela 模式
            vela_optimise=str(args.vela_optimise),  # 优化策略
            vela_silent=not bool(args.vela_verbose_log),  # 根据 verbose 参数决定是否静默
            vela_limit=args.vela_limit,  # Vela 子网数量上限
            rep_dataset_samples=int(args.vela_rep_dataset_samples),  # 量化代表性样本数
            quantize_int8=not bool(args.vela_float32),  # 若未指定 float32 则默认 INT8
            keep_artifacts=bool(args.vela_keep_artifacts),  # 是否保留中间产物
        )
        for row in records:  # 遍历所有记录
            sample_index = int(row["sample_index"])  # 当前子网索引
            if sample_index not in vela_map:  # 若超出 Vela 限制
                row["vela_status"] = "skipped_limit"  # 标记为因限制而跳过
                continue
            row.update(vela_map[sample_index])  # 合并 Vela 指标到记录中

    ranked = sorted(records, key=lambda item: (float(item["epe"]), int(item["sample_index"])))  # 按 EPE 升序排序（EPE 越小越好）
    ranking_rows: List[Dict[str, Any]] = []  # 用于写入 ranking CSV 的行
    for rank_idx, row in enumerate(ranked, start=1):  # 生成带 rank 序号的列表
        ranking_rows.append(
            {
                "rank": int(rank_idx),  # 排名
                "sample_index": int(row["sample_index"]),  # 对应采样索引
                "arch_code": row["arch_code"],  # 架构编码
                "epe": float(row["epe"]),  # EPE
                "complexity_score": float(row["complexity_score"]),  # 复杂度
            }
        )

    top_k = max(1, int(args.top_k))  # 保证 top_k 至少为 1
    vela_ok_rows = [item for item in records if str(item.get("vela_status", "")) == "ok"]  # 只选 Vela 运行成功的记录
    rank_fps = sorted(vela_ok_rows, key=lambda item: (-float(item["fps"]), int(item["sample_index"])))  # 按 FPS 从高到低排序
    rank_sram = sorted(vela_ok_rows, key=lambda item: (float(item["sram_peak_mb"]), int(item["sample_index"])))  # 按 SRAM 从小到大排序
    rank_epe = ranked  # 按 EPE 排序的完整列表

    summary_payload: Dict[str, Any] = {
        "status": "ok",  # 运行状态
        "checkpoint_type": args.checkpoint_type,  # 使用的 checkpoint 类型
        "checkpoint_path": str(checkpoint_prefix),  # checkpoint 实际路径
        "checkpoint_meta": _load_checkpoint_meta(path_prefix=checkpoint_prefix),  # checkpoint 元信息
        "num_arch_samples_requested": int(args.num_arch_samples),  # 请求的采样数
        "num_arch_samples_used": int(len(arch_pool)),  # 实际采样数
        "sample_seed": int(sample_seed),  # 子网采样随机种子
        "include_eval_pool": bool(include_eval_pool),  # 是否包含固定评估池
        "eval_pool_size": int(pool_size),  # 固定评估池大小
        "bn_recal_batches": int(bn_recal_batches),  # BN 重校准 batch 数
        "eval_batches_per_arch": int(eval_batches_per_arch),  # 每个子网评估 batch 数
        "batch_size": int(batch_size),  # 评估 batch 大小
        "num_workers_requested": int(args.num_workers),  # 命令行请求的 worker 数
        "num_workers_used": int(metrics.get("num_workers_used", 1)),  # 实际使用的 worker 数
        "vela_enabled": bool(args.enable_vela),  # 是否开启 Vela
        "vela_mode": str(args.vela_mode),  # Vela 模式
        "vela_optimise": str(args.vela_optimise),  # Vela 优化策略
        "vela_limit": args.vela_limit,  # Vela 子网数限制
        "distribution": {  # 各种指标的统计分布
            "epe": _compute_metric_summary([item.get("epe") for item in records]),  # EPE 分布
            "fps": _compute_metric_summary([item.get("fps") for item in records]),  # FPS 分布
            "sram_peak_mb": _compute_metric_summary([item.get("sram_peak_mb") for item in records]),  # SRAM 分布
            "inference_ms": _compute_metric_summary([item.get("inference_ms") for item in records]),  # 推理时间分布
        },
        "vela_status_count": {  # Vela 状态计数统计
            "ok": int(sum(1 for item in records if item.get("vela_status") == "ok")),  # 成功数量
            "fail": int(sum(1 for item in records if item.get("vela_status") == "fail")),  # 失败数量
            "skipped": int(sum(1 for item in records if item.get("vela_status") == "skipped")),  # 完全未跑 Vela 的数量
            "skipped_limit": int(sum(1 for item in records if item.get("vela_status") == "skipped_limit")),  # 因 Vela 限制而被跳过的数量
        },
        "top_k_by_epe": rank_epe[:top_k],  # EPE 最好的 top K
        "bottom_k_by_epe": rank_epe[-top_k:],  # EPE 最差的 bottom K
        "top_k_by_fps": rank_fps[:top_k],  # FPS 最高的 top K
        "top_k_by_sram": rank_sram[:top_k],  # SRAM 最小的 top K
    }

    records_csv_path = analysis_dir / "records.csv"  # 记录明细 CSV 路径
    ranking_csv_path = analysis_dir / "ranking_by_epe.csv"  # EPE 排名 CSV 路径
    vela_csv_path = analysis_dir / "vela_metrics.csv"  # Vela 指标 CSV 路径
    summary_json_path = analysis_dir / "summary.json"  # 汇总统计 JSON 路径
    pool_json_path = analysis_dir / "sampled_arch_pool.json"  # 采样架构池 JSON 路径

    _save_records_csv(path=records_csv_path, records=records)  # 写出记录 CSV
    _save_ranking_csv(path=ranking_csv_path, ranking=ranking_rows)  # 写出 EPE 排名 CSV
    _save_vela_csv(path=vela_csv_path, records=records)  # 写出 Vela 指标 CSV
    write_json(str(summary_json_path), summary_payload)  # 写出汇总 JSON
    write_json(
        str(pool_json_path),
        {
            "sample_seed": int(sample_seed),  # 采样随机种子
            "num_arch_samples": int(len(arch_pool)),  # 真实采样数
            "arch_pool": arch_pool,  # 采样的架构池
            "source_eval_pool_path": str(eval_pool_info.get("pool_path", "")),  # 固定评估池来源路径
        },
    )

    elapsed_seconds = float(time.perf_counter() - start_perf)  # 统计耗时（秒）
    finished_at = datetime.now(timezone.utc)  # 记录结束时间（UTC）
    result = {
        "status": "ok",  # 脚本运行状态
        "run_dir": str(run_dir),  # 本次运行根目录
        "analysis_dir": str(analysis_dir),  # 分析结果目录
        "summary_json": str(summary_json_path),  # 汇总 JSON 文件路径
        "records_csv": str(records_csv_path),  # 记录 CSV 路径
        "ranking_csv": str(ranking_csv_path),  # EPE 排名 CSV 路径
        "vela_csv": str(vela_csv_path),  # Vela 指标 CSV 路径
        "num_arch_samples_used": int(len(arch_pool)),  # 实际采样子网数量
        "mean_epe": float(summary_payload["distribution"]["epe"].get("mean", 0.0)),  # 全体子网 EPE 均值
        "mean_fps": float(summary_payload["distribution"]["fps"].get("mean", 0.0))
        if summary_payload["distribution"]["fps"].get("count_valid", 0) > 0
        else None,  # 若有有效 FPS 则给出均值，否则为 None
        "started_at_utc": started_at.isoformat(),  # 开始时间（ISO 格式）
        "finished_at_utc": finished_at.isoformat(),  # 结束时间（ISO 格式）
        "elapsed_seconds": elapsed_seconds,  # 运行耗时（秒）
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))  # 以 JSON 形式打印结果，保持中文不转义
    return 0  # 进程返回码 0 表示成功


if __name__ == "__main__":
    raise SystemExit(main())  # 当作为脚本执行时运行 main，并将返回码交给系统退出
