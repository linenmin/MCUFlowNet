"""Supernet 训练执行层实现。"""  # 定义模块用途
import csv  # 导入CSV模块
import random  # 导入随机模块
import subprocess  # 导入子进程模块
from pathlib import Path  # 导入路径工具
from typing import Any, Dict, List  # 导入类型注解

import numpy as np  # 导入NumPy模块
import tensorflow as tf  # 导入TensorFlow模块

from code.data.dataloader_builder import build_fc2_provider  # 导入FC2数据加载器构建函数
from code.data.transforms_180x240 import standardize_image_tensor  # 导入输入标准化函数
from code.engine.bn_recalibration import run_bn_recalibration_session  # 导入BN重估执行函数
from code.engine.checkpoint_manager import build_checkpoint_paths, find_existing_checkpoint, restore_checkpoint, save_checkpoint  # 导入checkpoint工具
from code.engine.early_stop import EarlyStopState, update_early_stop  # 导入早停工具
from code.engine.eval_step import build_epe_metric  # 导入EPE指标构建函数
from code.engine.train_step import add_weight_decay, build_multiscale_l1_loss  # 导入训练图构建函数
from code.nas.eval_pool_builder import build_eval_pool, check_eval_pool_coverage  # 导入验证池工具
from code.nas.fair_sampler import generate_fair_cycle  # 导入公平采样器
from code.network.MultiScaleResNet_supernet import MultiScaleResNetSupernet  # 导入超网模型
from code.optim.lr_scheduler import cosine_lr  # 导入学习率调度函数
from code.utils.json_io import write_json  # 导入JSON写入函数
from code.utils.logger import build_logger  # 导入日志构建函数
from code.utils.manifest import build_manifest  # 导入清单构建函数
from code.utils.path_utils import ensure_directory, project_root  # 导入路径工具
from code.utils.seed import set_global_seed  # 导入随机种子工具


def _resolve_output_dir(config: Dict[str, Any]) -> Path:  # 定义输出目录解析函数
    """解析实验输出目录并创建路径。"""  # 说明函数用途
    runtime_cfg = config.get("runtime", {})  # 读取运行配置字典
    output_root = runtime_cfg.get("output_root", "outputs/supernet")  # 读取输出根目录配置
    experiment_name = runtime_cfg.get("experiment_name", "default_experiment")  # 读取实验名称配置
    root_path = project_root() / output_root / experiment_name  # 计算实验输出绝对路径
    return ensure_directory(str(root_path))  # 创建并返回输出目录


def _resolve_resume_dir(config: Dict[str, Any], experiment_dir: Path) -> Path:  # 定义恢复目录解析函数
    """根据配置解析断点恢复来源目录。"""  # 说明函数用途
    runtime_cfg = config.get("runtime", {})  # 读取运行配置字典
    checkpoint_cfg = config.get("checkpoint", {})  # 读取断点配置字典
    resume_name = str(checkpoint_cfg.get("resume_experiment_name", "")).strip()  # 读取恢复实验名称
    if not resume_name:  # 判断是否显式指定恢复实验名称
        return experiment_dir  # 默认从当前实验目录恢复
    output_root = runtime_cfg.get("output_root", "outputs/supernet")  # 读取输出根目录配置
    return project_root() / output_root / resume_name  # 返回指定恢复实验目录


def _git_commit_hash() -> str:  # 定义提交哈希读取函数
    """获取当前仓库提交哈希。"""  # 说明函数用途
    try:  # 尝试读取Git哈希
        raw = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(project_root()))  # 执行Git命令
    except Exception:  # 捕获执行异常
        return "unknown"  # 返回未知哈希
    return raw.decode("utf-8").strip()  # 返回清洗后的哈希字符串


def _init_fairness_counts(num_blocks: int = 9) -> Dict[str, Dict[str, int]]:  # 定义公平计数初始化函数
    """初始化严格公平计数字典。"""  # 说明函数用途
    counts = {}  # 初始化计数字典
    for block_idx in range(num_blocks):  # 遍历每个选择块索引
        counts[str(block_idx)] = {"0": 0, "1": 0, "2": 0}  # 初始化三选项计数
    return counts  # 返回计数字典


def _sanitize_fairness_counts(raw_counts: Any, num_blocks: int = 9) -> Dict[str, Dict[str, int]]:  # 定义公平计数清洗函数
    """将外部计数字段清洗为规范结构。"""  # 说明函数用途
    clean = _init_fairness_counts(num_blocks=num_blocks)  # 初始化规范计数字典
    if not isinstance(raw_counts, dict):  # 判断输入结构是否为字典
        return clean  # 非字典时返回默认结构
    for block_idx in range(num_blocks):  # 遍历每个选择块索引
        block_key = str(block_idx)  # 计算字符串块索引
        block_raw = raw_counts.get(block_key, raw_counts.get(block_idx, {}))  # 读取块级计数字典
        if not isinstance(block_raw, dict):  # 判断块级结构是否合法
            continue  # 非法结构时跳过当前块
        for option in (0, 1, 2):  # 遍历三个选项
            option_key = str(option)  # 计算字符串选项键
            raw_value = block_raw.get(option_key, block_raw.get(option, 0))  # 读取原始计数值
            try:  # 尝试转换计数值为整数
                clean[block_key][option_key] = int(raw_value)  # 写入转换后的计数值
            except Exception:  # 捕获转换异常
                clean[block_key][option_key] = 0  # 异常时回退为零
    return clean  # 返回清洗后的计数字典


def _update_fairness_counts(counts: Dict[str, Dict[str, int]], cycle_codes: List[List[int]]) -> None:  # 定义公平计数更新函数
    """用单个公平周期更新选项计数。"""  # 说明函数用途
    for arch_code in cycle_codes:  # 遍历周期内3条路径编码
        for block_idx, option in enumerate(arch_code):  # 遍历每个块的选项
            counts[str(block_idx)][str(int(option))] += 1  # 累加对应选项计数


def _fairness_gap(counts: Dict[str, Dict[str, int]]) -> int:  # 定义公平差距计算函数
    """计算所有选择块的最大计数差。"""  # 说明函数用途
    gap = 0  # 初始化差距值
    for block_counts in counts.values():  # 遍历每个块计数字典
        values = list(block_counts.values())  # 获取三选项计数值
        gap = max(gap, max(values) - min(values))  # 更新最大差距
    return int(gap)  # 返回差距整数值


def _write_eval_history(csv_path: Path, rows: List[Dict[str, float]]) -> None:  # 定义评估历史写入函数
    """写入评估历史CSV文件。"""  # 说明函数用途
    headers = ["epoch", "mean_epe_12", "std_epe_12", "fairness_gap", "lr", "bn_recal_batches"]  # 定义CSV表头
    with csv_path.open("w", encoding="utf-8", newline="") as handle:  # 以UTF-8打开CSV文件
        writer = csv.DictWriter(handle, fieldnames=headers)  # 创建CSV写入器
        writer.writeheader()  # 写入CSV表头
        for row in rows:  # 遍历评估历史记录
            writer.writerow(row)  # 写入单行评估记录


def _build_graph(config: Dict[str, Any]) -> Dict[str, object]:  # 定义训练图构建函数
    """构建TF1超网训练图和关键张量。"""  # 说明函数用途
    train_cfg = config.get("train", {})  # 读取训练配置字典
    data_cfg = config.get("data", {})  # 读取数据配置字典
    batch_size = int(train_cfg.get("batch_size", 32))  # 读取批大小配置
    input_h = int(data_cfg.get("input_height", 180))  # 读取输入高度配置
    input_w = int(data_cfg.get("input_width", 240))  # 读取输入宽度配置
    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, input_h, input_w, 6], name="Input")  # 创建输入占位符
    label_ph = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, input_h, input_w, 2], name="Label")  # 创建标签占位符
    arch_code_ph = tf.compat.v1.placeholder(tf.int32, shape=[9], name="ArchCode")  # 创建架构编码占位符
    is_training_ph = tf.compat.v1.placeholder(tf.bool, shape=(), name="IsTraining")  # 创建训练标志占位符
    lr_ph = tf.compat.v1.placeholder(tf.float32, shape=(), name="LearningRate")  # 创建学习率占位符
    model = MultiScaleResNetSupernet(input_ph=input_ph, arch_code_ph=arch_code_ph, is_training_ph=is_training_ph, num_out=2, init_neurons=32, expansion_factor=2.0)  # 创建超网模型实例
    preds = model.build()  # 构建超网前向输出
    loss_tensor = build_multiscale_l1_loss(preds=preds, label_ph=label_ph)  # 构建多尺度L1损失
    loss_tensor = add_weight_decay(loss_tensor=loss_tensor, weight_decay=float(train_cfg.get("weight_decay", 0.0)))  # 叠加权重衰减项
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_ph, beta1=0.9, beta2=0.999, epsilon=1e-8)  # 创建基于lr占位符的优化器
    grads_and_vars = optimizer.compute_gradients(loss_tensor)  # 计算损失梯度变量对
    grads = [grad for grad, _ in grads_and_vars if grad is not None]  # 提取非空梯度张量
    vars_ = [var for grad, var in grads_and_vars if grad is not None]  # 提取对应变量列表
    clipped, global_norm = tf.clip_by_global_norm(grads, clip_norm=float(train_cfg.get("grad_clip_global_norm", 5.0)))  # 执行梯度裁剪
    accum_vars = []  # 初始化累积变量列表
    zero_ops = []  # 初始化清零操作列表
    add_ops = []  # 初始化累积操作列表
    for idx, (grad, var) in enumerate(zip(clipped, vars_)):  # 遍历裁剪梯度和变量
        accum_var = tf.compat.v1.get_variable(name=f"strict_accum_{idx}", shape=var.shape, dtype=var.dtype.base_dtype, initializer=tf.zeros_initializer(), trainable=False)  # 创建梯度累积变量
        accum_vars.append(accum_var)  # 记录累积变量
        zero_ops.append(tf.compat.v1.assign(accum_var, tf.zeros_like(accum_var), name=f"strict_zero_{idx}"))  # 创建清零操作
        add_ops.append(tf.compat.v1.assign_add(accum_var, grad, name=f"strict_add_{idx}"))  # 创建累积操作
    bn_updates = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)  # 获取BN更新操作集合
    with tf.control_dependencies(add_ops + bn_updates):  # 绑定累积与BN更新依赖
        accum_op = tf.no_op(name="strict_accum_done")  # 创建累积完成占位操作
    avg_grads = [accum_var / 3.0 for accum_var in accum_vars]  # 计算平均梯度列表
    with tf.control_dependencies([accum_op]):  # 绑定累积完成依赖
        apply_op = optimizer.apply_gradients(list(zip(avg_grads, vars_)), name="strict_apply")  # 创建梯度应用操作
    epe_tensor = build_epe_metric(pred_tensor=preds[-1], label_ph=label_ph)  # 构建最终尺度EPE指标
    saver = tf.compat.v1.train.Saver(max_to_keep=5)  # 创建Saver对象
    return {"input_ph": input_ph, "label_ph": label_ph, "arch_code_ph": arch_code_ph, "is_training_ph": is_training_ph, "lr_ph": lr_ph, "preds": preds, "loss": loss_tensor, "epe": epe_tensor, "global_grad_norm": global_norm, "zero_ops": zero_ops, "accum_op": accum_op, "apply_op": apply_op, "saver": saver}  # 返回图对象字典


def _run_eval_epoch(  # 定义评估轮执行函数
    sess,  # 定义TensorFlow会话参数
    graph_obj: Dict[str, object],  # 定义图对象字典参数
    train_provider,  # 定义训练数据采样器参数
    val_provider,  # 定义验证数据采样器参数
    eval_pool: List[List[int]],  # 定义验证架构池参数
    bn_recal_batches: int,  # 定义BN重估批次数参数
    batch_size: int,  # 定义批大小参数
) -> Dict[str, float]:  # 定义评估摘要返回类型
    """在固定验证子网池上执行评估。"""  # 说明函数用途
    epe_values = []  # 初始化EPE结果列表
    for arch_code in eval_pool:  # 遍历验证子网架构编码
        run_bn_recalibration_session(sess=sess, forward_fetch=graph_obj["preds"][-1], input_ph=graph_obj["input_ph"], label_ph=graph_obj["label_ph"], arch_code_ph=graph_obj["arch_code_ph"], is_training_ph=graph_obj["is_training_ph"], batch_provider=train_provider, arch_code=arch_code, batch_size=batch_size, recal_batches=bn_recal_batches)  # 执行BN统计重估
        input_batch, _, _, label_batch = val_provider.next_batch(batch_size=batch_size)  # 采样验证批数据
        input_batch = standardize_image_tensor(input_batch)  # 执行输入标准化
        epe_val = sess.run(graph_obj["epe"], feed_dict={graph_obj["input_ph"]: input_batch, graph_obj["label_ph"]: label_batch, graph_obj["arch_code_ph"]: arch_code, graph_obj["is_training_ph"]: False})  # 执行EPE推理
        epe_values.append(float(epe_val))  # 记录当前架构EPE
    mean_epe = float(np.mean(epe_values)) if epe_values else 0.0  # 计算平均EPE
    std_epe = float(np.std(epe_values)) if epe_values else 0.0  # 计算EPE标准差
    return {"mean_epe_12": mean_epe, "std_epe_12": std_epe}  # 返回评估摘要


def _try_restore_training_state(  # 定义训练状态恢复函数
    sess,  # 定义TensorFlow会话参数
    saver: tf.compat.v1.train.Saver,  # 定义Saver参数
    config: Dict[str, Any],  # 定义配置字典参数
    experiment_dir: Path,  # 定义当前实验目录参数
    logger,  # 定义日志器参数
) -> Dict[str, Any]:  # 定义恢复状态返回类型
    """按配置尝试恢复训练状态并返回恢复结果。"""  # 说明函数用途
    checkpoint_cfg = config.get("checkpoint", {})  # 读取断点配置字典
    if not bool(checkpoint_cfg.get("load_checkpoint", False)):  # 判断是否启用断点恢复
        return {"start_epoch": 1, "global_step": 0, "fairness_counts": _init_fairness_counts(num_blocks=9), "best_metric": float("inf"), "bad_epochs": 0}  # 返回默认初始状态
    resume_dir = _resolve_resume_dir(config=config, experiment_dir=experiment_dir)  # 解析恢复来源目录
    resume_paths = build_checkpoint_paths(str(resume_dir))  # 构建恢复目录checkpoint路径
    resume_prefix = find_existing_checkpoint(path_prefix=resume_paths["last"])  # 查找可恢复的last checkpoint
    if resume_prefix is None:  # 判断是否找到可恢复checkpoint
        logger.warning("load_checkpoint=true but no checkpoint found in %s", str(resume_dir))  # 记录恢复失败日志
        return {"start_epoch": 1, "global_step": 0, "fairness_counts": _init_fairness_counts(num_blocks=9), "best_metric": float("inf"), "bad_epochs": 0}  # 返回默认初始状态
    restore_info = restore_checkpoint(sess=sess, saver=saver, path_prefix=resume_prefix)  # 执行checkpoint恢复
    meta = restore_info.get("meta", {}) if isinstance(restore_info, dict) else {}  # 读取恢复meta字典
    start_epoch = int(meta.get("epoch", 0)) + 1  # 计算恢复后的起始轮数
    global_step = int(meta.get("global_step", 0))  # 读取恢复后的全局步数
    fairness_counts = _sanitize_fairness_counts(raw_counts=meta.get("fairness_counts", {}), num_blocks=9)  # 清洗恢复后的公平计数
    best_metric = float(meta.get("best_metric", float("inf")))  # 读取恢复后的最佳指标
    bad_epochs = int(meta.get("bad_epochs", 0))  # 读取恢复后的未提升轮数
    logger.info("resume checkpoint=%s start_epoch=%d global_step=%d", str(resume_prefix), start_epoch, global_step)  # 记录恢复成功日志
    return {"start_epoch": start_epoch, "global_step": global_step, "fairness_counts": fairness_counts, "best_metric": best_metric, "bad_epochs": bad_epochs}  # 返回恢复状态字典


def train_supernet(config: Dict[str, Any]) -> int:  # 定义超网训练主函数
    """执行Strict Fairness超网训练流程。"""  # 说明函数用途
    tf.compat.v1.disable_eager_execution()  # 关闭Eager执行模式
    tf.compat.v1.reset_default_graph()  # 重置默认计算图
    runtime_cfg = config.get("runtime", {})  # 读取运行配置字典
    train_cfg = config.get("train", {})  # 读取训练配置字典
    eval_cfg = config.get("eval", {})  # 读取评估配置字典
    data_cfg = config.get("data", {})  # 读取数据配置字典
    seed = int(runtime_cfg.get("seed", 42))  # 读取随机种子配置
    set_global_seed(seed)  # 设置全局随机种子
    experiment_dir = _resolve_output_dir(config)  # 解析输出目录
    logger = build_logger("edgeflownas_supernet", str(experiment_dir / "train.log"))  # 创建日志器
    logger.info("start strict-fairness supernet training")  # 记录训练开始日志
    batch_size = int(train_cfg.get("batch_size", 32))  # 读取批大小配置
    num_epochs = int(train_cfg.get("num_epochs", 200))  # 读取训练轮数配置
    steps_per_epoch = int(train_cfg.get("steps_per_epoch", 50))  # 读取每轮步数配置
    base_lr = float(train_cfg.get("lr", 1e-4))  # 读取基础学习率配置
    bn_recal_batches = int(eval_cfg.get("bn_recal_batches", 8))  # 读取BN重估批次数配置
    eval_pool_size = int(eval_cfg.get("eval_pool_size", 12))  # 读取验证池大小配置
    patience = int(eval_cfg.get("early_stop_patience", 15))  # 读取早停耐心配置
    min_delta = float(eval_cfg.get("early_stop_min_delta", 0.002))  # 读取早停改善阈值配置
    train_provider = build_fc2_provider(config=config, split_file_name=str(data_cfg.get("train_list_name", "FC2_train.txt")), seed_offset=0)  # 构建训练采样器
    val_provider = build_fc2_provider(config=config, split_file_name=str(data_cfg.get("val_list_name", "FC2_test.txt")), seed_offset=999)  # 构建验证采样器
    eval_pool = build_eval_pool(seed=seed, size=eval_pool_size, num_blocks=9)  # 构建固定验证池
    eval_pool_cov = check_eval_pool_coverage(pool=eval_pool, num_blocks=9)  # 执行验证池覆盖检查
    write_json(str(experiment_dir / f"eval_pool_{eval_pool_size}.json"), {"pool": eval_pool, "coverage": eval_pool_cov})  # 写入验证池文件
    graph_obj = _build_graph(config=config)  # 构建训练图对象
    early_stop = EarlyStopState()  # 初始化早停状态对象
    checkpoint_paths = build_checkpoint_paths(str(experiment_dir))  # 构建checkpoint路径
    eval_rows = []  # 初始化评估历史列表
    sampler_rng = random.Random(seed)  # 创建公平采样随机源
    total_steps = max(1, num_epochs * max(1, steps_per_epoch))  # 计算总训练步数
    with tf.compat.v1.Session() as sess:  # 创建TensorFlow会话
        sess.run(tf.compat.v1.global_variables_initializer())  # 初始化全局变量
        restore_state = _try_restore_training_state(sess=sess, saver=graph_obj["saver"], config=config, experiment_dir=experiment_dir, logger=logger)  # 尝试恢复训练状态
        start_epoch = int(restore_state["start_epoch"])  # 读取恢复后的起始轮数
        global_step = int(restore_state["global_step"])  # 读取恢复后的全局步数
        fairness_counts = _sanitize_fairness_counts(raw_counts=restore_state["fairness_counts"], num_blocks=9)  # 初始化公平计数字典
        early_stop.best_metric = float(restore_state["best_metric"])  # 恢复最佳指标状态
        early_stop.bad_epochs = int(restore_state["bad_epochs"])  # 恢复未提升轮数状态
        for epoch_idx in range(start_epoch, num_epochs + 1):  # 按轮数执行训练
            for _ in range(steps_per_epoch):  # 按每轮步数执行迭代
                cycle_codes = generate_fair_cycle(rng=sampler_rng, num_blocks=9)  # 生成当前公平周期编码
                _update_fairness_counts(counts=fairness_counts, cycle_codes=cycle_codes)  # 更新公平计数
                input_batch, _, _, label_batch = train_provider.next_batch(batch_size=batch_size)  # 采样训练批数据
                input_batch = standardize_image_tensor(input_batch)  # 执行输入标准化
                current_lr = cosine_lr(base_lr=base_lr, step_idx=global_step, total_steps=total_steps)  # 计算当前学习率
                sess.run(graph_obj["zero_ops"])  # 清零梯度累积变量
                for arch_code in cycle_codes:  # 遍历周期内3条路径
                    sess.run(graph_obj["accum_op"], feed_dict={graph_obj["input_ph"]: input_batch, graph_obj["label_ph"]: label_batch, graph_obj["arch_code_ph"]: arch_code, graph_obj["is_training_ph"]: True, graph_obj["lr_ph"]: current_lr})  # 执行单路径梯度累积
                loss_val, grad_norm_val, _ = sess.run([graph_obj["loss"], graph_obj["global_grad_norm"], graph_obj["apply_op"]], feed_dict={graph_obj["input_ph"]: input_batch, graph_obj["label_ph"]: label_batch, graph_obj["arch_code_ph"]: cycle_codes[0], graph_obj["is_training_ph"]: True, graph_obj["lr_ph"]: current_lr})  # 执行参数更新并抓取指标
                global_step += 1  # 递增全局步计数
            eval_info = _run_eval_epoch(sess=sess, graph_obj=graph_obj, train_provider=train_provider, val_provider=val_provider, eval_pool=eval_pool, bn_recal_batches=bn_recal_batches, batch_size=batch_size)  # 执行整轮评估
            row = {"epoch": int(epoch_idx), "mean_epe_12": float(eval_info["mean_epe_12"]), "std_epe_12": float(eval_info["std_epe_12"]), "fairness_gap": float(_fairness_gap(fairness_counts)), "lr": float(cosine_lr(base_lr=base_lr, step_idx=global_step, total_steps=total_steps)), "bn_recal_batches": float(bn_recal_batches)}  # 组装评估行记录
            eval_rows.append(row)  # 记录当前轮评估结果
            improved = update_early_stop(state=early_stop, metric=row["mean_epe_12"], min_delta=min_delta)  # 更新早停状态
            save_checkpoint(sess=sess, saver=graph_obj["saver"], path_prefix=checkpoint_paths["last"], epoch=epoch_idx, metric=row["mean_epe_12"], global_step=global_step, best_metric=early_stop.best_metric, bad_epochs=early_stop.bad_epochs, fairness_counts=fairness_counts, extra_payload={"row": row})  # 保存last checkpoint
            if improved:  # 判断是否刷新最佳指标
                save_checkpoint(sess=sess, saver=graph_obj["saver"], path_prefix=checkpoint_paths["best"], epoch=epoch_idx, metric=row["mean_epe_12"], global_step=global_step, best_metric=early_stop.best_metric, bad_epochs=early_stop.bad_epochs, fairness_counts=fairness_counts, extra_payload={"row": row})  # 保存best checkpoint
            logger.info("epoch=%d loss=%.6f mean_epe_12=%.6f std_epe_12=%.6f fairness_gap=%.2f grad_norm=%.4f", epoch_idx, float(loss_val), row["mean_epe_12"], row["std_epe_12"], row["fairness_gap"], float(grad_norm_val))  # 记录轮级日志
            if early_stop.bad_epochs >= patience:  # 判断是否满足早停条件
                logger.info("early stop triggered at epoch=%d", epoch_idx)  # 记录早停日志
                break  # 跳出训练循环
    _write_eval_history(csv_path=experiment_dir / "eval_epe_history.csv", rows=eval_rows)  # 写入评估历史文件
    write_json(str(experiment_dir / "fairness_counts.json"), fairness_counts)  # 写入公平计数文件
    manifest = build_manifest(config=config, git_commit=_git_commit_hash())  # 构建训练清单字典
    write_json(str(experiment_dir / "train_manifest.json"), manifest)  # 写入训练清单文件
    report_path = experiment_dir / "supernet_training_report.md"  # 计算训练报告路径
    report_path.write_text("# Supernet Training Report\n\n" f"- epochs_finished: {len(eval_rows)}\n" f"- best_metric: {early_stop.best_metric}\n" f"- final_fairness_gap: {_fairness_gap(fairness_counts)}\n" f"- eval_pool_coverage_ok: {bool(eval_pool_cov['ok'])}\n" f"- checkpoint_best: {checkpoint_paths['best']}\n" f"- checkpoint_last: {checkpoint_paths['last']}\n", encoding="utf-8")  # 写入训练报告内容
    logger.info("strict-fairness supernet training finished")  # 记录训练完成日志
    return 0  # 返回成功状态码
